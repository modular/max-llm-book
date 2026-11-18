# 4.3 Benchmarking kernels and layers

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/modular/max-llm-book/blob/main/notebooks/04_03_kernel_benchmarks.ipynb)
[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/modular/max-llm-book/blob/main/notebooks/04_03_kernel_benchmarks.ipynb)
[![View on GitHub](https://img.shields.io/badge/GitHub-notebook-black?logo=github)](https://github.com/modular/max-llm-book/blob/main/notebooks/04_03_kernel_benchmarks.ipynb)

A benchmark is an experiment, and most quick timings are bad experiments:
they include compilation, measure the time to *launch* work instead of the
time to *finish* it, or report one noisy sample. This section writes a small
benchmark harness that avoids those mistakes, then uses it to show why
decoding one token is limited by memory traffic and prefill by arithmetic.

<details class="code-fold">
<summary>Show the setup <span class="code-fold-lines">23 lines</span></summary>

```python
import logging
import time
import warnings

import numpy as np
import plotly.graph_objects as go
from max.driver import CPU
from max.dtype import DType
from max.graph import TensorType
from max.nn import Linear
from max.tensor import Tensor, default_device, default_dtype
from plotly.subplots import make_subplots

from minillm import viz
from minillm.device import pick_device
from minillm.config import LlamaConfig
from minillm.model import DecoderLayer
from minillm.weights import load_checkpoint, select

logging.getLogger("max").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="The eager interpreter failed")
DEVICE = pick_device()  # a GPU if MAX finds one, otherwise the CPU
config = LlamaConfig.from_pretrained()
```

</details>

## A harness

The harness follows four rules, in the spirit of MAX's layer benchmarks, which
time on the device with events and report the median and p95:

1. Warm up. The first calls compile kernels and fill caches, so don't time
   them.
2. Wait for the result. Work can run asynchronously, especially on a GPU.
   Reading the output (`to_numpy()`) forces it to finish.
3. Repeat. One sample is mostly noise, so take many.
4. Report the median and a tail percentile. Outliers pull the mean; the
   median shows the typical run and p90 shows how slow an unlucky one gets.

<details class="code-fold">
<summary>Show <code>Timing</code>, <code>benchmark</code> <span class="code-fold-lines">38 lines</span></summary>

```python
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass
class Timing:
    """Milliseconds per call over the measured iterations."""

    median_ms: float
    p90_ms: float
    min_ms: float

    def __str__(self) -> str:
        return f"median {self.median_ms:7.2f} ms  p90 {self.p90_ms:7.2f} ms"


def benchmark(
    fn: Callable[[], object], warmup: int = 3, iterations: int = 20
) -> Timing:
    """Times ``fn``, which must return a fully computed result (for example,
    a NumPy array read back from the device)."""
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000)
    return Timing(
        median_ms=float(np.median(samples)),
        p90_ms=float(np.percentile(samples, 90)),
        min_ms=float(np.min(samples)),
    )
```

</details>

<small>Saved to `minillm/bench.py`.</small>

## Memory-bound or compute-bound?

Every projection in the model is a matrix multiplication, `x @ W.T`, with
`x` holding one row per token. Its cost has two parts: reading `W` from
memory, which doesn't depend on the number of tokens, and the arithmetic,
which grows with it. For \\(n\\) tokens and a \\(K \\to N\\) projection in
`float32` (4 bytes per number), the ratio of the two is the operation's
*arithmetic intensity*:

\\[I = \\frac{\\text{FLOPs}}{\\text{bytes moved}} = \\frac{2nKN}{4\\,(KN + nK + nN)}\\]

A device can't run faster than its compute peak \\(P\_{\\text{peak}}\\), nor
faster than its memory bandwidth \\(B\\) can feed it, which gives the
*roofline*:

\\[P \\le \\min\\left(P\_{\\text{peak}},\\; B \\cdot I\\right)\\]

With few tokens, \\(I\\) is small and memory sets the limit; with many, \\(I\\)
grows until compute does. Measure SmolLM2's `gate_proj`
(\\(K = 576\\), \\(N = 1536\\)) for 1 to 2048 tokens:

<details class="code-fold">
<summary>Show the code <span class="code-fold-lines">29 lines</span></summary>

```python
with default_device(DEVICE), default_dtype(DType.float32):
    proj = Linear(config.hidden_size, config.intermediate_size, bias=False)
run = proj.compile(
    TensorType(DType.float32, ("tokens", config.hidden_size), device=DEVICE)
)

K, N = config.hidden_size, config.intermediate_size
rows = []
for tokens in [1, 4, 16, 64, 256, 1024, 2048]:
    x = Tensor(
        np.random.default_rng(0)
        .standard_normal((tokens, K))
        .astype(np.float32),
        device=DEVICE,
    )
    t = benchmark(lambda: run(x).to_numpy())
    flops = 2 * tokens * K * N
    bytes_moved = 4 * (K * N + tokens * (K + N))
    rows.append(
        (
            tokens,
            t.median_ms,
            flops / t.median_ms / 1e6,
            bytes_moved / t.median_ms / 1e6,
        )
    )
    print(
        f"{tokens:5} tokens  {t}  {rows[-1][2]:7.1f} GFLOP/s  {rows[-1][3]:6.1f} GB/s"
    )
```

</details>

```text
    1 tokens  median    0.56 ms  p90    0.63 ms      3.1 GFLOP/s     6.3 GB/s
    4 tokens  median    0.35 ms  p90    0.66 ms     20.3 GFLOP/s    10.2 GB/s
   16 tokens  median    0.39 ms  p90    0.47 ms     72.0 GFLOP/s     9.3 GB/s
   64 tokens  median    0.86 ms  p90    1.25 ms    131.0 GFLOP/s     4.7 GB/s
  256 tokens  median    2.07 ms  p90    2.97 ms    218.5 GFLOP/s     2.8 GB/s
 1024 tokens  median    8.22 ms  p90   11.20 ms    220.3 GFLOP/s     1.5 GB/s
 2048 tokens  median   10.09 ms  p90   14.24 ms    359.1 GFLOP/s     2.1 GB/s
```

<details class="code-fold">
<summary>Show the code <span class="code-fold-lines">82 lines</span></summary>

```python
tokens, ms, gflops, gbps = zip(*rows)
intensity = [
    2 * t * K * N / (4 * (K * N + t * (K + N))) for t in tokens
]  # FLOP per byte moved

fig = make_subplots(
    rows=1,
    cols=2,
    horizontal_spacing=0.14,
    subplot_titles=["Time per call", "Arithmetic throughput against intensity"],
)
fig.add_trace(
    go.Scatter(
        x=tokens,
        y=ms,
        mode="lines+markers",
        line_color=viz.BLUE,
        showlegend=False,
        hovertemplate="%{x} tokens: %{y:.3f} ms<extra></extra>",
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=intensity,
        y=gflops,
        mode="lines+markers+text",
        line_color=viz.ORANGE,
        text=[str(t) for t in tokens],
        textposition="top left",
        showlegend=False,
        customdata=tokens,
        hovertemplate="%{customdata} tokens<br>%{x:.1f} FLOP/byte<br>%{y:.0f} GFLOP/s<extra></extra>",
    ),
    row=1,
    col=2,
)
# The roof uses the best bandwidth and throughput this sweep reached, not the
# device's datasheet numbers, so every point sits on or under it.
peak, bandwidth = max(gflops), max(gbps)
roof_x = np.geomspace(min(intensity) / 2, max(intensity) * 2, 200)
fig.add_trace(
    go.Scatter(
        x=roof_x,
        y=np.minimum(peak, bandwidth * roof_x),
        mode="lines",
        line=dict(color=viz.INK_SECONDARY, dash="dash", width=1.5),
        name="measured roofline",
        hovertemplate="roof at %{x:.1f} FLOP/byte: %{y:.0f} GFLOP/s<extra></extra>",
    ),
    row=1,
    col=2,
)
fig.add_annotation(
    x=np.log10(peak / bandwidth),
    y=np.log10(peak),
    text=f"ridge: {peak / bandwidth:.0f} FLOP/byte",
    showarrow=True,
    arrowhead=0,
    ay=-30,
    row=1,
    col=2,
)
fig.update_xaxes(title_text="tokens", type="log", dtick=1, row=1, col=1)
fig.update_yaxes(title_text="ms (log scale)", type="log", dtick=1, row=1, col=1)
fig.update_xaxes(
    title_text="FLOP per byte moved (log scale)",
    type="log",
    dtick=1,
    row=1,
    col=2,
)
fig.update_yaxes(
    title_text="GFLOP/s (log scale)", type="log", dtick=1, row=1, col=2
)
fig.update_layout(
    title="gate_proj (576 to 1536) from 1 to 2048 tokens; labels are token counts",
    height=420,
    legend=dict(x=0.58, y=1.12),
)
viz.show(fig)
```

</details>

<div class="plotly-figure" data-src="generated/04_03_kernel_benchmarks_7_0.json"></div>

Read the two plots together, and hover over the points. The dashed line on the
right is the roofline drawn from this sweep: its slope is the best bandwidth any
point reached, its flat top the best throughput. Points left of the ridge sit
under the slanted, memory-bound part of the roof; points to the right sit under
the flat, compute-bound part. A device's datasheet ceilings are higher than
these measured ones.

With few tokens, the time barely grows: 1 to 16 tokens all take about half a
millisecond on this page's machine, an Apple-silicon Mac with a Metal GPU. Treat these
numbers as an example: yours depend on your hardware and vary from run to run.
Each weight is used for only a few multiply-adds after it is loaded, so
arithmetic throughput is low and the operation is *memory-bound*. At a matrix
this small, though, most of that time is a fixed cost per call, for launching
the work and reading the result back: the single-token call moves `W`'s 3.5 MB
at under 10 GB/s, far below the GPU's memory bandwidth. In a large model,
reading the weights dominates instead.

With many tokens, each weight, once loaded, is reused for every token, so
arithmetic throughput climbs and the time grows with the number of tokens.
The operation moves toward the *compute-bound* end, where the hardware's
arithmetic units set the limit.

Decoding one token per step runs every weight matrix at the memory-bound end.
That is why per-token latency depends on memory bandwidth, and why serving
engines batch many requests into each step
([Chapter 5](ch05.md)): here, 16 tokens cost barely
more than one. Prefill processes the whole prompt at once and runs toward the
compute-bound end. The handbook's
[GPU memory](https://handbook.modular.com/kernel-optimization/gpu-architecture-fundamentals/gpu-memory/)
page explains the same trade-off on GPUs.

## Benchmark a layer

The same harness measures a whole decoder layer at the shapes that matter
in serving: decode steps (one token) for several batch sizes, and prefills
of several lengths.

```python
with default_device(DEVICE), default_dtype(DType.float32):
    layer = DecoderLayer(config)
layer.load_state_dict(select(load_checkpoint(), "model.layers.0."))
run_layer = layer.compile(
    TensorType(
        DType.float32, ("batch", "seq_len", config.hidden_size), device=DEVICE
    )
)

print("shape (batch x seq)       time per call       tokens/s")
medians = {}
for batch, seq in [(1, 1), (8, 1), (32, 1), (1, 128), (1, 512)]:
    x = Tensor(
        np.random.default_rng(0)
        .standard_normal((batch, seq, config.hidden_size))
        .astype(np.float32),
        device=DEVICE,
    )
    t = benchmark(lambda: run_layer(x).to_numpy(), warmup=5, iterations=50)
    medians[batch, seq] = t.median_ms
    print(f"{batch:5} x {seq:<5}  {t}  {batch * seq / t.median_ms * 1000:9.0f}")
print(
    f"a decode step for 32 sequences costs {medians[32, 1] / medians[1, 1]:.1f}x one sequence's"
)
```

```text
shape (batch x seq)       time per call       tokens/s
    1 x 1      median    1.42 ms  p90    3.23 ms        704
    8 x 1      median    1.47 ms  p90    2.30 ms       5429
   32 x 1      median    1.68 ms  p90    2.35 ms      19027
    1 x 128    median    2.65 ms  p90    3.15 ms      48343
    1 x 512    median    8.16 ms  p90    8.84 ms      62743
a decode step for 32 sequences costs 1.2x one sequence's
```

<details class="code-fold">
<summary>Show the code <span class="code-fold-lines">32 lines</span></summary>

```python
labels = [f"{b} x {s}" for b, s in medians]
fig = make_subplots(
    rows=1,
    cols=2,
    horizontal_spacing=0.12,
    subplot_titles=["Time per call (ms)", "Tokens per second"],
)
fig.add_trace(
    go.Bar(
        x=labels,
        y=list(medians.values()),
        marker_color=viz.BLUE,
        showlegend=False,
        hovertemplate="%{x}: %{y:.2f} ms<extra></extra>",
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Bar(
        x=labels,
        y=[b * s / ms * 1000 for (b, s), ms in medians.items()],
        marker_color=viz.AQUA,
        showlegend=False,
        hovertemplate="%{x}: %{y:,.0f} tokens/s<extra></extra>",
    ),
    row=1,
    col=2,
)
fig.update_xaxes(title_text="batch x sequence length")
fig.update_layout(title="One decoder layer at serving shapes", height=400)
viz.show(fig)
```

</details>

<div class="plotly-figure" data-src="generated/04_03_kernel_benchmarks_10_0.json"></div>

On this machine, a decode step for 32 sequences costs well under twice a single
sequence's step, so throughput in tokens per second rises sharply with batch
size. Timings on a laptop are noisy, which is why the harness
reports a percentile alongside the median. A 512-token prefill runs through
one layer at tens of thousands of tokens per second; the model has 30 layers,
so a whole-model prefill would run at a thirtieth of that, under 2,000 tokens
per second.

These numbers describe this machine's GPU. On another CPU or GPU the curves have
the same shape at different scales, although the time isn't always monotonic in
size: the kernels behind each operation choose different strategies for
different shapes, so on some devices a smaller batch takes longer than a larger
one (this page's run doesn't show it). Measure the shapes you serve. MAX's own
[layer benchmarks](https://github.com/modular/modular/tree/main/max/python/layer_benchmarks)
run this kind of sweep for every layer, and its kernel benchmarks do the same
for individual kernels. For whole-server measurements,
[Section 7.2](07_02_benchmark_serving.md) uses
`max benchmark`. The handbook's
[LLM performance benchmarks](https://handbook.modular.com/inference-optimization/llm-performance-benchmarks/)
page explains how to read published numbers.

## Exercises

1. Run the matmul sweep for `down_proj` (1536 → 576). Where does it switch
   from memory-bound to compute-bound?
2. Benchmark the same layer with `iterations=3` five times. How much do the
   medians vary? How many iterations make the result stable?
3. Compare the eager and compiled layer at batch 1, sequence 1. Where does
   eager execution spend its time?

## Next

[Section 4.4](04_04_profiling.md) looks inside a run:
which operations take the time, using MAX's profiling tools.

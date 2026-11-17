# 4.4 Profiling

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/modular/max-llm-book/blob/main/notebooks/04_04_profiling.ipynb)
[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/modular/max-llm-book/blob/main/notebooks/04_04_profiling.ipynb)
[![View on GitHub](https://img.shields.io/badge/GitHub-notebook-black?logo=github)](https://github.com/modular/max-llm-book/blob/main/notebooks/04_04_profiling.ipynb)

A benchmark tells you *how long* something takes. A profile tells you
*where* the time goes, which is what you need before optimizing anything.
This section profiles at three levels: a whole generation run, the kernels
a compiled graph launches, and the parts of your own model.

<details class="code-fold">
<summary>Show the setup <span class="code-fold-lines">21 lines</span></summary>

```python
import logging
import warnings

import numpy as np
import plotly.graph_objects as go
from max.driver import CPU
from max.dtype import DType
from max.tensor import Tensor, default_device, default_dtype
from max.tensor.tensor import TensorType

from minillm import viz
from minillm.bench import benchmark
from minillm.device import pick_device
from minillm.config import LlamaConfig
from minillm.model import load_model
from minillm.weights import load_checkpoint

logging.getLogger("max").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="The eager interpreter failed")
DEVICE = pick_device()  # a GPU if MAX finds one, otherwise the CPU
config = LlamaConfig.from_pretrained()
```

</details>

## Profile a whole run

`max generate` takes a `--profile` flag. On an NVIDIA GPU with Nsight
Systems installed, it records a kernel trace and prints the kernels that
took the most GPU time. It always profiles the Python side with `cProfile`
too, which answers the first question to ask of any run: is the time in your
model, or in the code around it?

```python
!max generate --model HuggingFaceTB/SmolLM2-135M-Instruct \
    --quantization-encoding float32 --prompt "hello" --max-new-tokens 16 \
    --num-warmups 1 --profile 2>&1 | grep -A 8 "=== Top"
```

```text
=== Top 15 Python/CPU functions (93.9% of CPU time, 321.38 ms total) ===
     %       total     calls  function
 47.1%   151.33 ms        64  ~:0<built-in method numpy.from_dlpack>
 44.1%   141.86 ms        32  max/engine/api.py:118(_Model_execute)
  0.5%     1.45 ms        16  max/pipelines/lib/pipeline_variants/text_generation.py:597(execute)
  0.3%     1.05 ms        16  max/pipelines/sampling/sampling_logits_processor.py:109(_packed_rows_to_device)
  0.3%   844.17 us        16  max/pipelines/lib/module_registration.py:763(prepare_initial_token_inputs)
  0.2%   667.59 us        16  max/pipelines/lib/pipeline_variants/text_generation.py:610(_execute)
  0.2%   658.54 us        16  max/driver/buffer.py:143(inplace_copy_from)
```

The time is in the model: `_Model_execute`, the call into the compiled graph,
and, on a GPU, `numpy.from_dlpack`, which reads the logits back to the host and
so waits for the GPU to finish its work. That wait is model time too, a common
trap when reading profiles of asynchronous devices. On this page's machine, an
Apple-silicon Mac with a Metal GPU, the two take over 90% of the profiled time.
Treat these numbers as an example: yours depend on your hardware and vary from
run to run. The pipeline around the model (tokenization, scheduling, sampling,
and the KV cache bookkeeping) takes under a percent per function. For this
model, any speedup has to come from the compiled graph itself.

## See which kernels run

`MODULAR_DEBUG=op-log-level=TRACE` makes the runtime log every operation it
launches and completes. Trace the MLP from
[Section 1.4](01_04_building_blocks.md). The setting
must be in place when the process starts, so the code runs as a separate script:

```python
%%writefile trace_mlp.py
import numpy as np
from max.dtype import DType
from max.tensor import Tensor, default_device, default_dtype
from max.tensor.tensor import TensorType

from minillm.config import LlamaConfig
from minillm.device import pick_device
from minillm.layers import MLP

DEVICE = pick_device()
with default_device(DEVICE), default_dtype(DType.float32):
    mlp = MLP(LlamaConfig())
run = mlp.compile(TensorType(DType.float32, ("seq_len", 576), device=DEVICE))
print("--- run ---", flush=True)
run(Tensor(np.ones((4, 576), np.float32), device=DEVICE)).to_numpy()
```

```text
Overwriting trace_mlp.py
```

```python
!MODULAR_DEBUG=op-log-level=TRACE python trace_mlp.py 2>&1 | sed -n '/--- run ---/,$p' | grep LAUNCH | grep -v 'LAUNCH matmul[^ ]'
```

```text
[OP] LAUNCH matmul [id=22] target=gpu:0
[OP] LAUNCH Epilogue(matmul, Elementwise(silu, mul), operandIndices=[0]) [id=24] target=gpu:0
[OP] LAUNCH matmul [id=26] target=gpu:0
```

Each matmul logs a second `LAUNCH` line from inside the kernel, whose name MAX
joins to the graph operation's name with no space (`matmulmo.matmul`); the
`grep -v` drops those, which leaves one line per kernel the graph launches.

The MLP has five operations in your Python (three matmuls, `silu`, and a
multiply), more in the traced graph you saw in
[Section 3.2](03_02_graph_api.md), and three kernels
at run time. The `Epilogue` entry is the
second matmul with `silu` and `mul` fused into it: the activation is applied as
each tile of the matmul's output is produced, before it is written to memory.
That fusion is why the hand-written SwiGLU kernel in
[Section 3.1](03_01_custom_ops.md) had little left to
fuse, and the trace is how you'd find out. The handbook's
[kernel optimization for LLM inference](https://handbook.modular.com/kernel-optimization/kernel-optimization-for-llm-inference/)
page explains fusion and why it matters most on GPUs.

## Profile by decomposition

To find where time goes inside your own model, time its parts separately with
the harness from
[Section 4.3](04_03_kernel_benchmarks.md). Compile one
layer's attention, MLP, and norm, and the LM head, and time each at two shapes:
one decode step (1 token) and a prefill (256 tokens). Each time is multiplied by
how often the part runs in the model: 30 attention and MLP blocks, and 61 norms
(two per layer plus the final one).

<details class="code-fold">
<summary>Show the code <span class="code-fold-lines">31 lines</span></summary>

```python
model = load_model(config, load_checkpoint(), DEVICE)
hidden = TensorType(
    DType.float32, ("batch", "seq_len", config.hidden_size), device=DEVICE
)
layer = model.model.layers[0]
parts = {
    "attention (x30)": (layer.self_attn.compile(hidden), 30),
    "MLP (x30)": (layer.mlp.compile(hidden), 30),
    "norms (x61)": (layer.input_layernorm.compile(hidden), 61),
    "LM head": (model.lm_head.compile(hidden), 1),
}

shares, all_times = {}, {}
for tokens in [1, 256]:
    x = Tensor(
        np.random.default_rng(0)
        .standard_normal((1, tokens, config.hidden_size))
        .astype(np.float32),
        device=DEVICE,
    )
    times = {
        name: benchmark(lambda: run(x).to_numpy(), iterations=30).median_ms
        * count
        for name, (run, count) in parts.items()
    }
    total = sum(times.values())
    shares[tokens] = {name: t / total for name, t in times.items()}
    all_times[tokens] = times
    print(f"{tokens} token(s): {total:.1f} ms in total")
    for name, t in times.items():
        print(f"  {name:16} {t:7.2f} ms  {t / total:5.0%}")
```

</details>

```text
1 token(s): 139.5 ms in total
  attention (x30)    56.17 ms    40%
  MLP (x30)          30.12 ms    22%
  norms (x61)        50.82 ms    36%
  LM head             2.42 ms     2%
256 token(s): 290.6 ms in total
  attention (x30)   118.63 ms    41%
  MLP (x30)          68.72 ms    24%
  norms (x61)        63.74 ms    22%
  LM head            39.53 ms    14%
```

<details class="code-fold">
<summary>Show the code <span class="code-fold-lines">25 lines</span></summary>

```python
rows = ["decode (1 token)", "prefill (256 tokens)"]
fig = go.Figure(
    [
        go.Bar(
            y=rows,
            x=[shares[1][name], shares[256][name]],
            name=name,
            orientation="h",
            marker=dict(color=color, line=dict(color=viz.SURFACE, width=2)),
            customdata=[all_times[1][name], all_times[256][name]],
            hovertemplate=name
            + ": %{x:.0%} (%{customdata:.2f} ms)<extra>%{y}</extra>",
        )
        for name, color in zip(parts, viz.SERIES)
    ]
)
fig.update_layout(
    title="Where the model's time goes",
    barmode="stack",
    height=300,
    legend_traceorder="normal",
    xaxis=dict(title="share of model time", tickformat=".0%", range=[0, 1]),
    yaxis=dict(autorange="reversed", showgrid=False),
)
viz.show(fig)
```

</details>

<div class="plotly-figure" data-src="generated/04_04_profiling_10_0.json"></div>

Compare the two bars. The shares below were measured on this page's
Metal GPU.

Attention takes the largest share at both shapes, about 40%. Its cost is the
only one that grows with the square of the sequence length, so its share
grows on longer prompts; at 256 tokens the prompt is still short enough that
it barely moves.

The LM head's share grows the most, from a few percent to over a tenth. It
holds more weights than all 30 layers' attention put together (49,152 × 576),
and here it runs on every one of the 256 prompt positions. The cached model from
[Section 1.8](01_08_kv_cache.md) computes it for the
last position only, which keeps it small during prefill.

The norms look expensive at decode, over a quarter of the time, but that is
mostly an artifact of this method. Each separately compiled call pays a fixed
cost to launch its work and read back the result, and the decomposition counts
that cost 61 times for the norms; the whole compiled model runs as one graph and
pays it once. Time the whole model as one compiled graph to see how large that
overhead is:

```python
whole = model.compile(
    TensorType(DType.int64, ("batch", "seq_len"), device=DEVICE)
)
for tokens in [1, 256]:
    ids = Tensor(np.zeros((1, tokens), np.int64), device=DEVICE)
    whole_ms = benchmark(lambda: whole(ids).to_numpy(), iterations=30).median_ms
    parts_ms = sum(all_times[tokens].values())
    print(
        f"{tokens:3} token(s): whole model {whole_ms:6.1f} ms,"
        f" sum of parts {parts_ms:6.1f} ms ({parts_ms / whole_ms:.1f}x)"
    )
```

```text
  1 token(s): whole model   58.3 ms, sum of parts  139.5 ms (2.4x)
256 token(s): whole model  119.8 ms, sum of parts  290.6 ms (2.4x)
```

At both shapes the whole model takes less than half the time its parts add up
to. The difference is what the 122 separate calls cost: each one launches its
work and reads its result back to the host, and the whole graph does both once. So a
decomposition ranks the parts, but its shares overstate small operations such as
the norms. Check a surprising number with the kernel trace before acting on it.

Every optimization starts here: find the largest share, decide whether it is
memory-bound or compute-bound
([Section 4.3](04_03_kernel_benchmarks.md)), then look
at its kernels (the trace above).

## With Nsight Systems

On NVIDIA GPUs, Nsight Systems answers the same questions: set
`MODULAR_ENABLE_PROFILING=detailed` and run your program under `nsys profile`,
or pass `--profile` to `max generate`. `max benchmark --trace` does the same
for a server that already runs under `nsys launch`, started with
`--gpu-profiling detailed`. MAX marks its operations as
named ranges in the timeline, so you can match GPU kernels to the operations
that launched them. MAX's guide to
[GPU profiling with Nsight Systems](https://max.modular.com/gpu-system-profiling)
covers the setup. The handbook's
[kernel optimization tools](https://handbook.modular.com/kernel-optimization/kernel-optimization-tools/)
page surveys the tools.

## Exercises

1. Trace a full `DecoderLayer` and count its kernels. Which operations are fused
   into which?
2. Add a third shape, a decode step for a batch of 32, to the decomposition.
   Which part's share changes most?
3. Profile the decode step of the `KVCacheDecoder` from
   [Section 1.8](01_08_kv_cache.md) with `cProfile`
   from Python. How much of the step is Python overhead at this model size?

## Next

[Chapter 5](ch05.md) uses what you measured here to
build the core of a serving engine that batches many requests into each step.

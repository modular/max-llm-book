# 3.1 Your first custom op

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/modular/max-llm-book/blob/main/notebooks/03_01_custom_ops.ipynb)
[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/modular/max-llm-book/blob/main/notebooks/03_01_custom_ops.ipynb)
[![View on GitHub](https://img.shields.io/badge/GitHub-notebook-black?logo=github)](https://github.com/modular/max-llm-book/blob/main/notebooks/03_01_custom_ops.ipynb)

MAX's operations, from `F.silu` to flash attention, run kernels written in Mojo.
You can write your own the same way, compile it from a notebook cell, and call
it from Python like any other MAX operation. This section writes a fused SwiGLU
kernel, the `silu(gate) * up` inside SmolLM2's MLP from
[Section 1.4](01_04_building_blocks.md), tests it,
runs it inside your full model, and measures whether it helps.

<details class="code-fold">
<summary>Show the setup <span class="code-fold-lines">25 lines</span></summary>

```python
import logging
import time
import warnings
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from max import functional as F
from max.driver import CPU
from max.dtype import DType
from max.graph import TensorType
from max.pipelines import AutoTokenizer
from max.tensor import Tensor, default_device, default_dtype
from plotly.subplots import make_subplots

from minillm import viz
from minillm.device import pick_device
from minillm.config import MODEL_ID, LlamaConfig
from minillm.layers import MLP
from minillm.model import load_model
from minillm.weights import load_checkpoint

logging.getLogger("max").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="The eager interpreter failed")
DEVICE = pick_device()  # a GPU if MAX finds one, otherwise the CPU
```

</details>

## Write the kernel

`import mojo.notebook` adds a `%%mojo` cell magic to Jupyter. A cell that
starts with `%%mojo package -o kernels.mojoc` compiles its Mojo code into a
package that MAX can load.

```python
import mojo.notebook
```

The simplest kind of custom op is *elementwise*: each output element depends
only on the input elements at the same position. You write a struct with one
static `elementwise` method that maps input vectors to an output vector, and
MAX handles everything else: splitting the work across threads, loading and
storing memory, and choosing CPU or GPU code.

- `@extensibility.register("swiglu")` names the op; Python calls it by
  this name.
- `ElementwiseBinaryOp` declares the shape of the method: two inputs, one
  output, all of one dtype.
- `SIMD[dtype, width]` is a vector of `width` values. MAX picks the width that
  fills the hardware's vector registers, so one call processes several
  elements at once.
- `comptime assert` runs at compile time and rejects integer dtypes, for
  which `exp` is undefined.

SiLU is \\(x \\cdot \\sigma(x) = x / (1 + e^{-x})\\):

```mojo
%%mojo package -o kernels.mojoc
import extensibility
from extensibility import ElementwiseBinaryOp
from std.math import exp


@extensibility.register("swiglu")
struct SwiGLU(ElementwiseBinaryOp):
    @staticmethod
    def elementwise[
        dtype: DType, width: SIMDLength
    ](gate: SIMD[dtype, width], up: SIMD[dtype, width]) -> SIMD[dtype, width]:
        comptime assert dtype.is_floating_point(), "SwiGLU needs floats"
        return gate / (1 + exp(-gate)) * up
```

## Call it from Python

`F.custom` calls a custom op by name on eager tensors. It needs the device to
run on, the inputs, the type of each output (here, the same as `gate`'s), and
the package to load the kernel from:

```python
KERNELS = Path("kernels.mojoc").resolve()


def swiglu(gate: Tensor, up: Tensor) -> Tensor:
    return F.custom(
        "swiglu",
        device=gate.device,
        values=[gate, up],
        out_types=[gate.type],
        custom_extensions=KERNELS,
    )[0]


rng = np.random.default_rng(0)
g = rng.standard_normal((4, 8)).astype(np.float32)
u = rng.standard_normal((4, 8)).astype(np.float32)
result = swiglu(Tensor(g, device=DEVICE), Tensor(u, device=DEVICE))
print(result.to_numpy()[0, :4])
print(F.silu(Tensor(g, device=DEVICE)).to_numpy()[0, :4] * u[0, :4])
```

```text
[-0.01063813 -0.03336792  0.09002377  0.01961606]
[-0.01063813 -0.03336792  0.09002377  0.01961606]
```

## Test it

A kernel needs the same tests as any other code, against a reference you
trust. Check shapes that don't divide evenly into vectors, a single element,
and inputs extreme enough to overflow `exp`:

```python
def reference(g: np.ndarray, u: np.ndarray) -> np.ndarray:
    return g / (1 + np.exp(-g.astype(np.float64))) * u


cases = {
    "1 element": (1,),
    "odd length": (1, 7, 1001),
    "model size": (2, 32, 1536),
}
for name, shape in cases.items():
    g = rng.standard_normal(shape).astype(np.float32) * 4
    u = rng.standard_normal(shape).astype(np.float32)
    out = swiglu(Tensor(g, device=DEVICE), Tensor(u, device=DEVICE)).to_numpy()
    print(f"{name:12} max error {np.abs(out - reference(g, u)).max():.1e}")
    assert np.allclose(out, reference(g, u), rtol=1e-4, atol=1e-5)

extremes = np.array([-100.0, -20.0, 0.0, 20.0, 100.0], np.float32)
out = swiglu(
    Tensor(extremes, device=DEVICE),
    Tensor(np.ones(5, np.float32), device=DEVICE),
)
print("extremes:", out.to_numpy())
assert np.isfinite(out.to_numpy()).all()
```

```text
1 element    max error 1.0e-08
odd length   max error 1.8e-06
model size   max error 3.0e-06
extremes: [-0.0000000e+00 -4.1223068e-08  0.0000000e+00  2.0000000e+01
  1.0000000e+02]
```

Sweep the input to see the whole function and its error. With `up` set to
ones, the op is SiLU itself:

```python
sweep = np.linspace(-12, 12, 481, dtype=np.float32)
ones = np.ones_like(sweep)
ours = swiglu(
    Tensor(sweep, device=DEVICE), Tensor(ones, device=DEVICE)
).to_numpy()
error = np.abs(ours - reference(sweep, ones))
```

<details class="code-fold">
<summary>Show the code <span class="code-fold-lines">34 lines</span></summary>

```python
fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    row_heights=[0.6, 0.4],
)
fig.add_trace(
    go.Scatter(
        x=sweep,
        y=ours,
        name="your kernel",
        line_color=viz.BLUE,
        hovertemplate="silu(%{x:.2f}) = %{y:.4f}<extra></extra>",
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=sweep,
        y=np.maximum(error, 1e-12),
        name="error against float64 NumPy",
        line_color=viz.ORANGE,
        hovertemplate="%{x:.2f}: error %{y:.1e}<extra></extra>",
    ),
    row=2,
    col=1,
)
fig.update_yaxes(title_text="output", row=1, col=1)
fig.update_yaxes(title_text="absolute error", type="log", dtick=1, row=2, col=1)
fig.update_xaxes(title_text="gate", row=2, col=1)
fig.update_layout(title="The swiglu op with up = 1", height=520)
viz.show(fig)
```

</details>

<div class="plotly-figure" data-src="generated/03_01_custom_ops_13_0.json"></div>

The error stays at the scale of `float32` rounding, about \\(10^{-7}\\) of the
output or less, across the whole range.

At \\(-100\\), \\(e^{100}\\) overflows `float32` to infinity, and \\(-100 / \\infty\\) is
\\(-0\\), the correct limit. A different but equivalent formula, such as
`gate * exp(gate) / (1 + exp(gate))`, would compute \\(\\infty / \\infty\\) and
return NaN at \\(+100\\). Tests with extreme inputs catch this kind of bug.

## Use it in your model

`FusedMLP` is the MLP from
[Section 1.4](01_04_building_blocks.md) with the
custom op in place of `F.silu(...) *`:

```python
class FusedMLP(MLP):
    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


config = LlamaConfig.from_pretrained()
weights = load_checkpoint()
model = load_model(config, weights, DEVICE)
for layer in model.model.layers:
    with F.lazy(), default_device(DEVICE), default_dtype(DType.float32):
        fused = FusedMLP(config)
    fused.load_state_dict(dict(layer.mlp.parameters))
    layer.mlp = fused
```

Compile the model as before, with one addition: `custom_extensions` tells
the compiler where to find the kernel. Then compare the logits with the
model from [Section 1.6](01_06_full_model.md):

```python
token_type = TensorType(DType.int64, ("batch", "seq_len"), device=DEVICE)
fused_model = model.compile(token_type, custom_extensions=[KERNELS])
plain_model = load_model(config, weights, DEVICE).compile(token_type)

tokenizer = AutoTokenizer(MODEL_ID)
ids = Tensor(
    np.array([tokenizer(["The capital of France is"])["input_ids"][0]]),
    device=DEVICE,
)
fused_logits = fused_model(ids).to_numpy()
plain_logits = plain_model(ids).to_numpy()
print(f"max difference {np.abs(fused_logits - plain_logits).max():.1e}")
print(
    "next token:", repr(tokenizer.decode([int(fused_logits[0, -1].argmax())]))
)
assert (
    np.abs(fused_logits - plain_logits).max() < 1e-2
    and fused_logits[0, -1].argmax() == plain_logits[0, -1].argmax()
)
```

```text
max difference 4.9e-05
next token: ' Paris'
```

## Is it faster?

Time a compiled MLP block with and without the custom op, on a batch of 512
positions:

<details class="code-fold">
<summary>Show the code <span class="code-fold-lines">28 lines</span></summary>

```python
with default_device(DEVICE), default_dtype(DType.float32):
    plain_mlp = MLP(config)
    fused_mlp = FusedMLP(config)
fused_mlp.load_state_dict(dict(plain_mlp.parameters))

block_type = TensorType(
    DType.float32, ("batch", "seq_len", config.hidden_size), device=DEVICE
)
runs = {
    "MAX ops": plain_mlp.compile(block_type),
    "custom op": fused_mlp.compile(block_type, custom_extensions=[KERNELS]),
}
x = Tensor(
    rng.standard_normal((1, 512, config.hidden_size)).astype(np.float32),
    device=DEVICE,
)
for run in runs.values():
    for _ in range(3):
        run(x).to_numpy()
times = {name: [] for name in runs}
# Alternate the two, so a change in clock speed or load hits both alike.
for _ in range(30):
    for name, run in runs.items():
        start = time.perf_counter()
        run(x).to_numpy()
        times[name].append((time.perf_counter() - start) * 1000)
for name, ms in times.items():
    print(f"{name:12} median {np.median(ms):.2f} ms")
```

</details>

```text
MAX ops      median 3.58 ms
custom op    median 3.41 ms
```

<details class="code-fold">
<summary>Show the code <span class="code-fold-lines">21 lines</span></summary>

```python
fig = go.Figure(
    [
        go.Box(
            y=ms,
            name=name,
            marker_color=color,
            boxpoints="all",
            jitter=0.4,
            pointpos=0,
            hovertemplate="%{y:.2f} ms<extra>" + name + "</extra>",
        )
        for (name, ms), color in zip(times.items(), [viz.BLUE, viz.ORANGE])
    ]
)
fig.update_layout(
    title="A compiled MLP block on 512 positions, 30 runs each",
    height=400,
    yaxis=dict(title="ms", rangemode="tozero"),
    showlegend=False,
)
viz.show(fig)
```

</details>

<div class="plotly-figure" data-src="generated/03_01_custom_ops_21_0.json"></div>

In the runs behind this page, on an Apple-silicon Mac with a Metal GPU, the two
boxes overlap almost completely, the medians land within about 10% of each
other, and which one is faster changes from run to run. Treat the output above
as one example: your hardware will give different numbers, so measure on the
hardware you serve on. There is
little to gain because the graph compiler already fuses elementwise operations
into their neighbors: the kernel trace in
[Section 4.4](04_04_profiling.md) shows `silu` and the
multiply applied in the epilogue of a matmul, so the plain version never writes
the activation to memory either. Most of the time goes to the three matrix
multiplications.

Write a custom op when MAX doesn't have the operation you need, or when a better
*algorithm* needs control that a chain of separate operations can't express.
Flash attention ([Section 1.5](01_05_attention.md)) is
the classic case: it never materializes the attention matrix, which no
combination of separate operations can avoid. Elementwise ops written as above
stay fusible, so the compiler can still merge them with their neighbors, and the
same kernel runs on a GPU with no changes.

Kernels that need control over threads and memory, such as a matmul or
attention, implement an `execute` method instead, written with Mojo's GPU
programming model. MAX's guide to
[building custom ops](https://max.modular.com/develop/build-custom-ops) shows
the pattern, and the
[Mojo GPU puzzles](https://github.com/modular/mojo-gpu-puzzles) teach GPU
programming in Mojo from first principles. The handbook's
[kernel optimization tools](https://handbook.modular.com/kernel-optimization/kernel-optimization-tools/)
page compares Mojo with CUDA and Triton, and
[GPU architecture fundamentals](https://handbook.modular.com/kernel-optimization/gpu-architecture-fundamentals/)
explains what a kernel has to work with.

## Exercises

1. Write a `gelu_tanh` custom op as an `ElementwiseUnaryOp` (unary
   elementwise ops take one input) and test it against `F.gelu` with
   `approximate="tanh"`.
2. Add a `float16` case to the tests. How large is the error, and why?
3. Fuse more: write an op that takes `gate`, `up`, and a scale, and returns
   `silu(gate) * up * scale`. Does the timing change?

## Next

[Section 3.2](03_02_graph_api.md) looks underneath
`compile`: the graph MAX builds from your module, and the graph API you can use
to build one directly.

# 1.2 Tensors and eager execution

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/modular/max-llm-book/blob/main/notebooks/01_02_tensors_and_eager.ipynb)
[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/modular/max-llm-book/blob/main/notebooks/01_02_tensors_and_eager.ipynb)
[![View on GitHub](https://img.shields.io/badge/GitHub-notebook-black?logo=github)](https://github.com/modular/max-llm-book/blob/main/notebooks/01_02_tensors_and_eager.ipynb)

Every model in this book is written with three MAX modules:

- `max.tensor`: the `Tensor` type.
- `max.functional` (imported as `F`): operations on tensors, such as
  `F.softmax` and `F.silu`.
- `max.nn`: `Module` and layers such as `Linear`, built from the two above.

This section covers what you need from each: creating tensors, running
operations, and moving data to and from NumPy. It also shows how MAX differs
from NumPy. Operations run eagerly, and the same code also compiles into a
fast graph.

```python
import logging
import time

import numpy as np
import plotly.graph_objects as go
from max import functional as F
from max.driver import CPU, accelerator_count
from max.dtype import DType
from max.nn import Linear, Module
from max.tensor import Tensor, default_device, default_dtype
from max.tensor.tensor import TensorType

from minillm import viz

logging.getLogger("max").setLevel(logging.ERROR)
```

## Pick a device

MAX runs the same code on CPUs and on NVIDIA, AMD, and Apple silicon GPUs.
`max.tensor.defaults()` returns the dtype and the device mesh MAX would
choose on its own, and `pick_device` takes the mesh's first device: a GPU if
MAX finds one, otherwise the CPU. Every notebook in this book uses that
device, so on Colab or Kaggle with a GPU runtime, or on a Mac, your code
runs on the GPU without changes.

Code cells that start with `# export:` become part of this book's `minillm`
package, so later sections import them instead of repeating them.
`pick_device` is the first:

```python
from __future__ import annotations

from max.driver import Device
from max.tensor import defaults


def pick_device() -> Device:
    """Returns the device MAX uses by default: a GPU if there is one,
    otherwise the CPU."""
    _, mesh = defaults()
    return mesh.devices[0]
```

<small>Saved to `minillm/device.py`.</small>

```python
DEVICE = pick_device()
print("GPUs found:", accelerator_count(), " using:", DEVICE)
```

```text
GPUs found: 1  using: Device(type=gpu,id=0)
```

## Create tensors

A `Tensor` has a shape, a dtype, and a device. You can build one from Python
lists, from a NumPy array, or with a constructor such as `Tensor.ones`. The
examples name the dtype, `float32`, for a reason the next cells show:

```python
x = Tensor(
    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=DType.float32, device=DEVICE
)
print(x)
print(x.shape, x.dtype, x.device)
```

```text
Tensor([1 2 3
 4 5 6], dtype=DType.float32, device=Device(type=gpu,id=0))
[Dim(2), Dim(3)] DType.float32 Device(type=gpu,id=0)
```

```python
print(Tensor(np.arange(6, dtype=np.float32).reshape(2, 3), device=DEVICE))
print(Tensor.zeros([2, 2], dtype=DType.float32, device=DEVICE))
print(Tensor.arange(5, dtype=DType.float32, device=DEVICE))
```

```text
Tensor([0 1 2
 3 4 5], dtype=DType.float32, device=Device(type=gpu,id=0))
Tensor([0 0
 0 0], dtype=DType.float32, device=Device(type=gpu,id=0))
Tensor([0 1 2 3 4], dtype=DType.float32, device=Device(type=gpu,id=0))
```

Without `device=` or `dtype=`, MAX also picks the dtype: `bfloat16` on a GPU, a
16-bit format most models ship in, and `float32` on a CPU. This book computes in
`float32` everywhere, so its results can be compared exactly with reference
implementations; the exercises in
[Section 4.2](04_02_logit_verification.md) measure
what `bfloat16` changes. NumPy also has no `bfloat16` type, so a GPU tensor left
at the default dtype can't be read with `to_numpy()`.

```python
print("default placement:", Tensor.ones([2]).device, Tensor.ones([2]).dtype)
```

```text
default placement: Device(type=gpu,id=0) DType.bfloat16
```

To change the defaults for a block of code, use the `default_device` and
`default_dtype` context managers. Later sections build models inside them.

```python
with default_device(DEVICE), default_dtype(DType.float32):
    y = Tensor.ones([2, 2])
print(y.device, y.dtype)
```

```text
Device(type=gpu,id=0) DType.float32
```

## Run operations

Python operators work as in NumPy, including `@` for matrix multiplication
and `.T` to transpose the last two axes. `max.functional` holds everything
else.

```python
print(x @ x.T)
print(x.sum(axis=-1))
print(x[:, 1:])
print(x.reshape([3, 2]))
print(F.softmax(x, axis=-1))
```

```text
Tensor([14 32
 32 77], dtype=DType.float32, device=Device(type=gpu,id=0))
Tensor([ 6
 15], dtype=DType.float32, device=Device(type=gpu,id=0))
Tensor([2 3
 5 6], dtype=DType.float32, device=Device(type=gpu,id=0))
Tensor([1 2
 3 4
 5 6], dtype=DType.float32, device=Device(type=gpu,id=0))
Tensor([0.09003  0.2447  0.6652
 0.09003  0.2447  0.6652], dtype=DType.float32, device=Device(type=gpu,id=0))
```

Results are tensors on the same device as their inputs. To read one as a
NumPy array, call `to_numpy()`, which copies it from the GPU first if
needed. For a tensor already in CPU memory, `np.from_dlpack()` shares the
memory instead of copying:

```python
print(x.to_numpy())
on_cpu = x.to(CPU())
print(np.from_dlpack(on_cpu).dtype)
```

```text
[[1. 2. 3.]
 [4. 5. 6.]]
float32
```

## Eager execution

Each operation above ran immediately, so you could print its result. MAX
builds a small graph for each operation, compiles it into a kernel the first
time it sees that operation and shape, and caches the kernel. Compare the
first call with the later ones:

```python
z = Tensor(np.random.rand(256, 256).astype(np.float32), device=DEVICE)
call_ms = []
for i in range(4):
    start = time.perf_counter()
    F.gelu(z @ z).to_numpy()
    call_ms.append((time.perf_counter() - start) * 1000)
    print(f"call {i}: {call_ms[-1]:.0f} ms")
```

```text
call 0: 10106 ms
call 1: 19 ms
call 2: 6 ms
call 3: 5 ms
```

<details class="code-fold">
<summary>Show the code <span class="code-fold-lines">14 lines</span></summary>

```python
fig = go.Figure(
    go.Bar(
        x=[f"call {i}" for i in range(4)],
        y=call_ms,
        marker_color=[viz.ORANGE] + [viz.BLUE] * 3,
        hovertemplate="%{x}: %{y:.1f} ms<extra></extra>",
    )
)
fig.update_layout(
    title="F.gelu(z @ z): the first call compiles, later calls reuse the kernels",
    yaxis=dict(title="ms (log scale)", type="log", dtick=1),
    height=360,
)
viz.show(fig)
```

</details>

<div class="plotly-figure" data-src="generated/01_02_tensors_and_eager_19_0.json"></div>

The first call pays for compilation. On this page's machine, an Apple-silicon
Mac with a Metal GPU, it took several seconds, and the later calls took a few
milliseconds. Treat these numbers as an example: yours depend on your hardware
and vary from run to run. The cache persists across sessions, and
`max warm-interpreter-cache` compiles the common kernels ahead of time.

### Defer work with `F.lazy`

Inside `F.lazy()`, operations record a graph without running it. The
tensors stay *unrealized* until you read a value:

```python
with F.lazy():
    u = F.silu(x) * 2
print(u)
print(u.to_numpy())
```

```text
Tensor(
    '<unrealized>',
    shape=[Dim(2), Dim(3)],
    dtype=float32,
    device=Device(type=gpu,id=0)
)
[[ 1.4621172  3.523188   5.7154446]
 [ 7.85611    9.933072  11.970329 ]]
```

Later sections build models inside `F.lazy()`, so the random initial weights
are never computed: real weights from the checkpoint replace them.

### Fuse operations with `@F.functional`

A function decorated with `@F.functional` runs as one graph, so MAX can
fuse its operations into fewer kernels:

```python
@F.functional
def silu_times_two(t: Tensor) -> Tensor:
    return F.silu(t) * 2


print(silu_times_two(x))
```

```text
Tensor([1.462 3.523 5.715
 7.856 9.933 11.97], dtype=DType.float32, device=Device(type=gpu,id=0))
```

## Modules and compilation

A `Module` groups parameters (tensors) and a `forward` method. Here is the
feed-forward block SmolLM2 uses, at its real size: hidden size 576, intermediate
size 1536. [Section 1.4](01_04_building_blocks.md)
explains the design; for now it's a realistic workload.

```python
class MLP(Module):
    def __init__(self, hidden: int, intermediate: int) -> None:
        self.gate_proj = Linear(hidden, intermediate, bias=False)
        self.up_proj = Linear(hidden, intermediate, bias=False)
        self.down_proj = Linear(intermediate, hidden, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


with default_device(DEVICE), default_dtype(DType.float32):
    mlp = MLP(576, 1536)
print(mlp)
print([name for name, _ in mlp.parameters])
```

```text
MLP(
    gate_proj=Linear(in_dim=Dim(576), out_dim=Dim(1536), bias=False),
    up_proj=Linear(in_dim=Dim(576), out_dim=Dim(1536), bias=False),
    down_proj=Linear(in_dim=Dim(1536), out_dim=Dim(576), bias=False)
)
['gate_proj.weight', 'up_proj.weight', 'down_proj.weight']
```

Calling the module runs it eagerly, one operation at a time. A single timing
is noisy, so `median_ms` calls the function 20 times, after one untimed call
that compiles its kernels, and reports the median:

```python
def median_ms(fn, x: Tensor, repeats: int = 20) -> float:
    fn(x).to_numpy()  # warm up: the first call compiles its kernels
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn(x).to_numpy()
        times.append((time.perf_counter() - start) * 1000)
    return float(np.median(times))


inputs = Tensor(np.random.rand(1, 64, 576).astype(np.float32), device=DEVICE)
eager_out = mlp(inputs).to_numpy()
print(f"eager:    {median_ms(mlp, inputs):.1f} ms")
```

```text
eager:    46.9 ms
```

`compile` traces `forward` once into a whole-model graph and optimizes it:
it fuses operations, plans memory, and removes the per-operation overhead.
`TensorType` declares the input. Named dimensions such as `"batch"` and
`"seq_len"` may take any size at run time.

```python
start = time.perf_counter()
compiled_mlp = mlp.compile(
    TensorType(DType.float32, ("batch", "seq_len", 576), device=DEVICE)
)
print(f"compiled in {time.perf_counter() - start:.1f} s")

compiled_out = compiled_mlp(inputs).to_numpy()
print(f"compiled: {median_ms(compiled_mlp, inputs):.1f} ms")
relative = np.abs(compiled_out - eager_out).max() / np.abs(eager_out).max()
print(f"max difference, relative to the largest output: {relative:.1e}")
assert relative < 1e-5
```

```text
compiled in 10.9 s
compiled: 1.2 ms
max difference, relative to the largest output: 9.7e-07
```

How much faster depends on the input size. Each eager operation pays a
fixed cost to dispatch, which dominates small inputs; the compiled graph
pays it once. Time both over a range of sequence lengths:

```python
lengths = [1, 16, 64, 256, 1024]
timings = {"eager": [], "compiled": []}
for length in lengths:
    x_len = Tensor(
        np.random.rand(1, length, 576).astype(np.float32), device=DEVICE
    )
    timings["eager"].append(median_ms(mlp, x_len))
    timings["compiled"].append(median_ms(compiled_mlp, x_len))
    print(
        f"{length:5} tokens: eager {timings['eager'][-1]:6.1f} ms, "
        f"compiled {timings['compiled'][-1]:6.1f} ms"
    )
```

```text
    1 tokens: eager   21.9 ms, compiled    0.5 ms
   16 tokens: eager   28.4 ms, compiled    1.3 ms
   64 tokens: eager   42.7 ms, compiled    1.6 ms
  256 tokens: eager   51.6 ms, compiled    3.9 ms
 1024 tokens: eager   42.5 ms, compiled    8.0 ms
```

<details class="code-fold">
<summary>Show the code <span class="code-fold-lines">20 lines</span></summary>

```python
fig = go.Figure(
    [
        go.Scatter(
            x=lengths,
            y=ms,
            name=name,
            mode="lines+markers",
            line_color=color,
            hovertemplate=name + ", %{x} tokens: %{y:.2f} ms<extra></extra>",
        )
        for (name, ms), color in zip(timings.items(), [viz.ORANGE, viz.BLUE])
    ]
)
fig.update_layout(
    title="The MLP block, eager and compiled (median of 20 calls)",
    height=420,
    xaxis=dict(title="sequence length (tokens)", type="log", dtick=1),
    yaxis=dict(title="ms (log scale)", type="log", dtick=1),
)
viz.show(fig)
```

</details>

<div class="plotly-figure" data-src="generated/01_02_tensors_and_eager_32_0.json"></div>

The compiled MLP gives the same numbers up to `float32` rounding (the compiled
graph may order additions differently) and runs many times faster. In the run
shown here, on an Apple-silicon Mac with a Metal GPU, an eager call took about
20 to 50 ms at every length, and a compiled one took about 1 ms up to 64 tokens
and a few milliseconds at the longest lengths. The rest of the book works the same way:
write and debug a model eagerly, printing anything you like, then compile it to
run it for real. MAX's docs cover
[eager execution](https://max.modular.com/develop/eager-execution) and
[tensor realization](https://max.modular.com/develop/tensor-realization) in more
detail.

The handbook's
[CPU vs GPU vs TPU](https://handbook.modular.com/llm-inference-basics/cpu-vs-gpu-vs-tpu/)
and
[GPU architecture fundamentals](https://handbook.modular.com/kernel-optimization/gpu-architecture-fundamentals/)
pages explain why matrix multiplications dominate the cost and why GPUs speed
them up.

## Exercises

1. Create a `[4, 4]` tensor, apply `F.softmax` along each axis, and check
   with NumPy that the rows (or columns) sum to 1.
2. Time the eager and compiled MLP with `seq_len` 1, 64, and 512. Where is
   the speedup largest? Why might small inputs benefit most?
3. Print a tensor created inside `F.lazy()` before and after calling
   `.to_numpy()` on it.

## Next

[Section 1.3](01_03_tokens_and_weights.md) loads
SmolLM2's real weights and turns token IDs into vectors.

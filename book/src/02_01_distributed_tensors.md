# 2.1 Distributed tensors

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/modular/max-llm-book/blob/main/notebooks/02_01_distributed_tensors.ipynb)
[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/modular/max-llm-book/blob/main/notebooks/02_01_distributed_tensors.ipynb)
[![View on GitHub](https://img.shields.io/badge/GitHub-notebook-black?logo=github)](https://github.com/modular/max-llm-book/blob/main/notebooks/02_01_distributed_tensors.ipynb)

A *distributed tensor* is one logical array whose data lives on several devices.
In MAX it is an ordinary `Tensor`, the same type you used from
[Section 1.2](01_02_tensors_and_eager.md) on:
operators, `max.functional`, and `max.nn` layers accept it, run on every device
that holds a piece, and add the communication between devices that the math
requires.

This section runs everything on a *simulated* mesh: four "devices" that are
all the same CPU. The program, the per-device pieces, and the results are
the same as on four GPUs, so you can develop and test distributed code on a
laptop and run it unchanged on real hardware.

```python
import logging
import warnings

import numpy as np
import plotly.graph_objects as go
from max import functional as F
from max.driver import CPU
from max.tensor import Tensor
from max.tensor.sharding import (
    DeviceMapping,
    DeviceMesh,
    Partial,
    Replicated,
    Sharded,
    auto_reshard,
    visualize,
)
from plotly.subplots import make_subplots

from minillm import viz

logging.getLogger("max").setLevel(logging.ERROR)
```

## Meshes and placements

A `DeviceMesh` arranges devices on named axes. This one has four devices on
one axis, named `tp` for tensor parallelism:

```python
mesh = DeviceMesh((CPU(),) * 4, (4,), ("tp",))


def on(*placements) -> DeviceMapping:
    """Places a tensor on the mesh, with one placement per mesh axis."""
    return DeviceMapping(mesh, placements)
```

A `DeviceMapping` says how a tensor's data relates to the devices on each
mesh axis. That relation is its *placement*:

- `Replicated()`: every device holds the whole tensor.
- `Sharded(axis)`: the tensor is split along `axis`, one piece per device.
- `Partial()`: every device holds a full-size tensor, and the real value is
  their sum.

Split an 8 × 4 matrix by rows:

```python
a = np.arange(32, dtype=np.float32).reshape(8, 4)
x = Tensor(a, device=on(Sharded(0)))
visualize(x)
```

```text
DType.float32 [8, 4] with DeviceMapping(DeviceMesh(tp=4), [Sharded(axis=0)])
┌──────────┐   tensor[:, :]
│ device 0 │   ├── tp=0  device 0 - tensor[:2, :]
│ [:2, :]  │   ├── tp=1  device 1 - tensor[2:4, :]
├──────────┤   ├── tp=2  device 2 - tensor[4:6, :]
│ device 1 │   └── tp=3  device 3 - tensor[6:, :]
│ [2:4, :] │
├──────────┤
│ device 2 │
│ [4:6, :] │
├──────────┤
│ device 3 │
│ [6:, :]  │
└──────────┘
```

`x.shape` is the global shape, `[8, 4]`, as if the tensor lived on one
device. Each device's piece, its *shard*, is in `local_shards`:

```python
print("global shape:", [int(d) for d in x.shape])
for i, shard in enumerate(x.local_shards):
    print(f"device {i}:", [int(d) for d in shard.shape], shard.to_numpy()[:, 0])
```

```text
global shape: [8, 4]
device 0: [2, 4] [0. 4.]
device 1: [2, 4] [ 8. 12.]
device 2: [2, 4] [16. 20.]
device 3: [2, 4] [24. 28.]
```

The next helper turns a tensor's placement into a map of which device holds
each element, using only `mapping.placements` and the shard shapes. The
figures in this section draw it: one color per device, gray where every
device holds the element, and a darker gray for a `Partial` tensor, whose
every device holds a term of the sum.

```python
ALL, PARTIAL = -1, -2


def owners(t: Tensor) -> np.ndarray:
    """Returns the device that holds each element of ``t`` on a 1-D mesh."""
    grid = np.full([int(d) for d in t.shape], ALL)
    placement = t.mapping.placements[0]
    if isinstance(placement, Partial):
        grid[:] = PARTIAL
    elif isinstance(placement, Sharded):
        start = 0
        for device, shard in enumerate(t.local_shards):
            size = int(shard.shape[placement.axis])
            index = [slice(None)] * grid.ndim
            index[placement.axis] = slice(start, start + size)
            grid[tuple(index)] = device
            start += size
    return grid


HOLDER = {ALL: "every device", PARTIAL: "every device, one term each"} | {
    d: f"device {d}" for d in range(4)
}
# One band per value from -2 to 3: partial, all, then devices 0 to 3.
BANDS = [
    viz.INK_SECONDARY,
    viz.GRID,
    viz.BLUE,
    viz.ORANGE,
    viz.AQUA,
    viz.YELLOW,
]
OWNER_SCALE = [
    [edge, color]
    for i, color in enumerate(BANDS)
    for edge in (i / 6, (i + 1) / 6)
]


def owner_heatmap(t: Tensor, show_values: bool = True, **trace) -> go.Heatmap:
    grid = owners(t)
    values = t.to_numpy()
    rows, cols = grid.shape
    holder = [
        [HOLDER[int(grid[r, c])] for c in range(cols)] for r in range(rows)
    ]
    return go.Heatmap(
        z=grid,
        zmin=-2.5,
        zmax=3.5,
        colorscale=OWNER_SCALE,
        showscale=False,
        xgap=2,
        ygap=2,
        customdata=holder,
        text=values.round(1) if show_values else None,
        texttemplate="%{text}" if show_values else None,
        hovertemplate="row %{y}, column %{x}<br>held by %{customdata}<extra></extra>",
        **trace,
    )


def device_legend(fig: go.Figure) -> None:
    for d in range(4):
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name=f"device {d}",
                marker=dict(symbol="square", size=12, color=BANDS[d + 2]),
            )
        )
```

Move the slider to compare placements of the same 8 × 4 matrix:

```python
placements = {
    "Sharded(0)": Sharded(0),
    "Sharded(1)": Sharded(1),
    "Replicated()": Replicated(),
}
steps = [[owner_heatmap(Tensor(a, device=on(p)))] for p in placements.values()]
fig = go.Figure(
    layout=dict(
        title="Which device holds each element of an 8 × 4 matrix",
        height=520,
        xaxis=dict(showgrid=False, dtick=1),
        yaxis=dict(showgrid=False, autorange="reversed", dtick=1),
    )
)
device_legend(fig)
viz.show(viz.with_slider(fig, steps, list(placements), "placement "))
```

<div class="plotly-figure" data-src="generated/02_01_distributed_tensors_12_0.json"></div>

`to_numpy()` gathers the shards back into one array, so checking a
distributed result against NumPy is one line:

```python
print(np.array_equal(x.to_numpy(), a))
```

```text
True
```

## Sharded matrix multiplication

Models spend most of their time in matrix multiplications, and there are two
natural ways to split \\(y = x W\\) across devices. Every tensor-parallel
transformer uses both.

In a *column-parallel* multiplication, every device holds all of \\(x\\) and a
slice of \\(W\\)'s columns, and computes the matching slice of \\(y\\). It needs no
communication:

```python
w = np.random.default_rng(0).standard_normal((4, 6)).astype(np.float32)

y = Tensor(a, device=on(Replicated())) @ Tensor(w, device=on(Sharded(1)))
print(y.mapping)
print("correct:", np.allclose(y.to_numpy(), a @ w, atol=1e-5))
assert np.allclose(y.to_numpy(), a @ w, atol=1e-5)
```

```text
DeviceMapping(DeviceMesh(tp=4), [Sharded(axis=1)])
correct: True
```

In a *row-parallel* multiplication, each device holds a slice of \\(x\\)'s
columns and the matching slice of \\(W\\)'s rows. Each device's product is one
term of the full sum over the inner dimension, so the result is `Partial`:

```python
z = Tensor(a, device=on(Sharded(1))) @ Tensor(w, device=on(Sharded(0)))
print(z.mapping)
print("correct:", np.allclose(z.to_numpy(), a @ w, atol=1e-5))
assert np.allclose(z.to_numpy(), a @ w, atol=1e-5)
```

```text
DeviceMapping(DeviceMesh(tp=4), [Partial()])
correct: True
```

A `Partial` tensor is correct only once its terms are summed. Most
operations, such as an activation, need the real value, so MAX must first
combine the shards with a *collective*, an operation that communicates
between devices. Inside `auto_reshard(mode="warn")`, MAX reports every
collective it inserts and the reason for it:

```python
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    with auto_reshard(mode="warn"):
        activated = F.relu(z)
print(next(w.message for w in caught if "reshards" in str(w.message)))
print("result:", activated.mapping)
```

```text

relu(x) reshards x
  x      : float32[8, 6]@mesh(tp=4){"tp":P}
  region : auto_reshard(DEFAULT_TRANSITIONS, mode="warn")

  decided per mesh axis
  axis    x          -> out       collective         bytes   status
  "tp"    x{P->S0}   -> S0        reduce_scatter(x)     144  not allowed
          x{P->S1}   -> S1        reduce_scatter(x)     144  not allowed
          x{P->R}    -> R         allreduce(x)          288  chosen, reshards x
  result  x{"tp":R}  -> {"tp":R}

  x = transfer_to(x, DeviceMapping(mesh, (R,)))
  with auto_reshard(mode="silent"):
  with auto_reshard(ALL_TRANSITIONS, mode="silent"):   # also allows the cheaper rows above
  at /var/folders/gg/plf_qcqs5556rt10q1s96gl40000gn/T/ipykernel_87578/1282956743.py:4 in <module>()
result: DeviceMapping(DeviceMesh(tp=4), [Replicated()])
```

MAX chose an *all-reduce*: every device sends its partial sum and receives
the total. It also considered a *reduce-scatter*, which sums and leaves each
device one slice, but that transition isn't in the default set. Outside an
`auto_reshard` block, MAX raises an error instead of inserting a collective,
so a model can't add communication you didn't plan for.

To move a tensor to a placement yourself, use `F.transfer_to`:

```python
replicated = F.transfer_to(z, on(Replicated()))
print(replicated.mapping)
```

```text
DeviceMapping(DeviceMesh(tp=4), [Replicated()])
```

## Column then row: one all-reduce per block

Chaining the two patterns is the core of *tensor parallelism*, as introduced
by the Megatron-LM paper. A column-parallel layer leaves its output split by
columns, which is the input a row-parallel layer needs, so the pair runs with
no communication between them and a single all-reduce at the end. Here is a
small MLP block, checked against NumPy:

```python
rng = np.random.default_rng(1)
w_up = rng.standard_normal((4, 16)).astype(np.float32)
w_down = rng.standard_normal((16, 4)).astype(np.float32)

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    with auto_reshard(mode="warn"):
        h = F.relu(
            Tensor(a, device=on(Replicated()))
            @ Tensor(w_up, device=on(Sharded(1)))
        )
        partial_out = h @ Tensor(w_down, device=on(Sharded(0)))
reshards = [w for w in caught if "reshards" in str(w.message)]
out = F.transfer_to(partial_out, on(Replicated()))  # the one all-reduce

print("hidden:", h.mapping, " block output:", partial_out.mapping)
print("collectives MAX had to insert:", len(reshards))
print(
    "correct:",
    np.allclose(out.to_numpy(), np.maximum(a @ w_up, 0) @ w_down, atol=1e-3),
)
assert not reshards and np.allclose(
    out.to_numpy(), np.maximum(a @ w_up, 0) @ w_down, atol=1e-3
)
```

```text
hidden: DeviceMapping(DeviceMesh(tp=4), [Sharded(axis=1)])  block output: DeviceMapping(DeviceMesh(tp=4), [Partial()])
collectives MAX had to insert: 0
correct: True
```

Here is the whole block as placements. Every device holds all of `x`. The
column-parallel weight and the hidden activation are split by columns in the
same way, so each device multiplies only its own slice. The row-parallel
weight is split by rows to match, and the block's output, `h W_down`, is a
partial sum on every device (dark gray) until the all-reduce sums it:

```python
panels = {
    "x": Tensor(a, device=on(Replicated())),
    "W_up": Tensor(w_up, device=on(Sharded(1))),
    "h": h,
    "W_down": Tensor(w_down, device=on(Sharded(0))),
    "h W_down": partial_out,
    "summed": out,
}
widths = [int(t.shape[1]) for t in panels.values()]
fig = make_subplots(
    rows=1,
    cols=len(panels),
    column_widths=widths,
    horizontal_spacing=0.03,
    subplot_titles=list(panels),
)
for col, t in enumerate(panels.values(), start=1):
    fig.add_trace(owner_heatmap(t, show_values=False), row=1, col=col)
device_legend(fig)
fig.update_xaxes(showgrid=False, showticklabels=False)
fig.update_yaxes(showgrid=False, showticklabels=False, autorange="reversed")
fig.update_layout(
    title="A column-parallel, row-parallel MLP block on 4 devices",
    height=420,
    legend=dict(y=-0.04, yanchor="top"),
    margin=dict(t=96),
)
viz.show(fig)
```

<div class="plotly-figure" data-src="generated/02_01_distributed_tensors_26_0.json"></div>

The hidden activation stayed sharded, `relu` ran on each shard independently,
and the only communication was the explicit all-reduce at the end.
[Section 2.2](02_02_tensor_parallel.md) applies this
pattern to every attention and MLP block of SmolLM2.

## Two-dimensional meshes

Real deployments combine strategies. A mesh with two axes can split a batch
of requests across `dp` (data parallelism) while splitting each layer's
weights across `tp`. Each mesh axis gets its own placement:

```python
mesh_2d = DeviceMesh((CPU(),) * 4, (2, 2), ("dp", "tp"))
batch = Tensor(
    np.ones((4, 6), np.float32),
    device=DeviceMapping(mesh_2d, (Sharded(0), Sharded(1))),
)
visualize(batch)
```

```text
DType.float32 [4, 6] with DeviceMapping(DeviceMesh(dp=2, tp=2), [Sharded(axis=0), Sharded(axis=1)])
┌──────────┬──────────┐   tensor[:, :]
│ device 0 │ device 1 │   ├── dp=0 - tensor[:2, :]
│ [:2, :3] │ [:2, 3:] │   │   ├── tp=0  device 0 - tensor[:2, :3]
├──────────┼──────────┤   │   └── tp=1  device 1 - tensor[:2, 3:]
│ device 2 │ device 3 │   └── dp=1 - tensor[2:, :]
│ [2:, :3] │ [2:, 3:] │       ├── tp=0  device 2 - tensor[2:, :3]
└──────────┴──────────┘       └── tp=1  device 3 - tensor[2:, 3:]
```

On real hardware, the cost of each collective depends on how much data it moves
and how fast the devices are connected, and that decides the best layout for a
given model and cluster. The handbook's
[parallelism strategies](https://handbook.modular.com/inference-optimization/data-tensor-pipeline-expert-hybrid-parallelism/)
page compares data, tensor, pipeline, and expert parallelism. MAX's docs cover
[distributed tensors](https://max.modular.com/develop/distributed/tensors) in
more detail.

## Exercises

1. Shard `x` with `Sharded(1)` instead of `Sharded(0)` and visualize it.
   What happens with 3 devices instead of 4, which don't divide 4 columns
   evenly?
2. Swap the order in the MLP block: row-parallel first, then column-parallel.
   Which collectives does MAX insert now?
3. Count the bytes each device sends in the all-reduce of the MLP block.
   How does it scale with the batch size and the hidden size?

## Next

[Section 2.2](02_02_tensor_parallel.md) splits SmolLM2
across devices and checks that its logits don't change.

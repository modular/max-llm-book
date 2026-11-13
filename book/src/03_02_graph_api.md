# 3.2 Under the hood: the graph API

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/modular/max-llm-book/blob/main/notebooks/03_02_graph_api.ipynb)
[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/modular/max-llm-book/blob/main/notebooks/03_02_graph_api.ipynb)
[![View on GitHub](https://img.shields.io/badge/GitHub-notebook-black?logo=github)](https://github.com/modular/max-llm-book/blob/main/notebooks/03_02_graph_api.ipynb)

Since [Section 1.2](01_02_tensors_and_eager.md) you
have called `compile` on modules and gotten a fast callable back. This section
opens that box. `compile` traces your module into a *graph*, a complete
description of the computation, and MAX's compiler turns the graph into
optimized code. You can also build graphs directly with `max.graph`, which is
how many of MAX's serving pipelines build their models, and load checkpoints
with MAX's production weight loader.

```python
import logging
import re
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from huggingface_hub import hf_hub_download
from max.driver import CPU, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, Weight, ops
from max.graph.weights import load_weights
from max.tensor import Tensor, default_device, default_dtype

from minillm import viz
from minillm.device import pick_device
from minillm.config import MODEL_ID, LlamaConfig
from minillm.layers import MLP
from minillm.weights import load_checkpoint, select

logging.getLogger("max").setLevel(logging.ERROR)
DEVICE = pick_device()  # a GPU if MAX finds one, otherwise the CPU

config = LlamaConfig.from_pretrained()
weights = load_checkpoint()
```

## What `compile` builds

`trace` runs the first half of `compile`: it calls `forward` with symbolic
inputs and records every operation into a graph, without compiling it. Here is
SmolLM2's MLP from
[Section 1.4](01_04_building_blocks.md):

```python
with default_device(DEVICE), default_dtype(DType.float32):
    mlp = MLP(config)
mlp.load_state_dict(select(weights, "model.layers.0.mlp."))

staged = mlp.trace(
    TensorType(
        DType.float32,
        ("seq_len", config.hidden_size),
        device=DeviceRef.from_device(DEVICE),
    )
)
print("\n".join(str(staged.graph).splitlines()[:12]))
```

```text
mo.graph @MLP<seq_len>(%arg0: !mo.tensor<[seq_len, 576], f32, gpu:0>) -> !mo.tensor<[seq_len, 576], f32, gpu:0> attributes {_kernel_library_paths = [], argument_names = ["input0"], result_names = ["output0"]} {
  %0 = mo.chain.create()
  %1 = mo.constant.external [name = "gate_proj.weight", align = 1, device = #M.device_ref<"cpu", 0>] : !mo.tensor<[1536, 576], f32>
  %result, %outChain = mo.transfer[%0] %1 : !mo.tensor<[1536, 576], f32> to <"gpu", 0>
  %2 = mo.constant <#M.dense_array<1, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %3 = mo.transpose(%result, %2) : (!mo.tensor<[1536, 576], f32, gpu:0>, !mo.tensor<[2], si64>) -> !mo.tensor<[576, 1536], f32, gpu:0>
  %4 = rmo.matmul(%arg0, %3) : (!mo.tensor<[seq_len, 576], f32, gpu:0>, !mo.tensor<[576, 1536], f32, gpu:0>) -> !mo.tensor<[seq_len, 1536], f32, gpu:0>
  %5 = mo.constant <#M.dense_array<0.000000e+00> : tensor<f32>> : !mo.tensor<[], f32, gpu:0>
  %6 = rmo.add(%4, %5) : (!mo.tensor<[seq_len, 1536], f32, gpu:0>, !mo.tensor<[], f32, gpu:0>) -> !mo.tensor<[seq_len, 1536], f32, gpu:0>
  %7 = mo.silu(%6) : !mo.tensor<[seq_len, 1536], f32, gpu:0>
  %8 = mo.constant.external [name = "up_proj.weight", align = 1, device = #M.device_ref<"cpu", 0>] : !mo.tensor<[1536, 576], f32>
  %result_0, %outChain_1 = mo.transfer[%0] %8 : !mo.tensor<[1536, 576], f32> to <"gpu", 0>
```

The text is a dataflow graph: each line defines one value (`%4`) from
earlier ones. Parse it and draw it, with each operation placed one column
after the latest value it reads. Hover over a node to see its line of IR:

```python
LINE = re.compile(
    r"^\s*(%\w+)(?:, %\w+)* = ([\w.]+)(.*)$"
)  # some ops have two results
nodes, edges = {"%arg0": ("input", "input0: [seq_len, 576]")}, []
for line in str(staged.graph).splitlines():
    match = LINE.match(line)
    if not match or match.group(2) in ("mo.chain.create", "mo.constant"):
        continue  # skip bookkeeping and small constants
    value, op, rest = match.groups()
    weight = re.search(r'name = "([^"]+)"', rest)
    label = weight.group(1) if weight else op.split(".")[-1]
    nodes[value] = (label, line.strip())
    edges += [(src, value) for src in re.findall(r"%\w+", rest) if src in nodes]
output = re.search(r"mo.output (%\w+)", str(staged.graph)).group(1)
nodes["output"] = ("output", "output0: [seq_len, 576]")
edges.append((output, "output"))

depth = {}
for value in nodes:  # lines are in dataflow order
    depth[value] = max(
        (depth[src] + 1 for src, dst in edges if dst == value), default=0
    )
column = {}
position = {}
for value in nodes:
    row = column.get(depth[value], 0)
    column[depth[value]] = row + 1
    position[value] = (depth[value], -row)


def kind(label: str) -> str:
    if label in ("input", "output"):
        return "input or output"
    if label.endswith(".weight"):
        return "weight"
    if label == "transfer":
        return "copy to device"
    return "matmul" if label == "matmul" else "other operation"


colors = {
    "input or output": viz.VIOLET,
    "weight": viz.ORANGE,
    "copy to device": viz.YELLOW,
    "matmul": viz.BLUE,
    "other operation": viz.AQUA,
}
edge_x, edge_y = [], []
for src, dst in edges:
    edge_x += [position[src][0], position[dst][0], None]
    edge_y += [position[src][1], position[dst][1], None]
fig = go.Figure(
    go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(color=viz.GRID, width=2),
        hoverinfo="skip",
        showlegend=False,
    )
)
for name, color in colors.items():
    members = [v for v in nodes if kind(nodes[v][0]) == name]
    fig.add_trace(
        go.Scatter(
            x=[position[v][0] for v in members],
            y=[position[v][1] for v in members],
            mode="markers+text",
            name=name,
            text=[nodes[v][0] for v in members],
            textposition="top center",
            marker=dict(
                size=18, color=color, line=dict(color=viz.SURFACE, width=2)
            ),
            customdata=[nodes[v][1][:110] for v in members],
            hovertemplate="%{customdata}<extra></extra>",
        )
    )
fig.update_layout(
    title="The MLP's graph, as compile sees it",
    height=480,
    xaxis=dict(visible=False, range=[-0.8, max(depth.values()) + 0.6]),
    yaxis=dict(visible=False, range=[-max(column.values()) + 0.5, 0.6]),
)
viz.show(fig)
```

<div class="plotly-figure" data-src="generated/03_02_graph_api_6_0.json"></div>

The graph is MAX's intermediate representation (in MLIR). A few things to
read in it:

- The input type is `[seq_len, 576]`: `seq_len` stays symbolic, which is why
  one compiled graph serves every sequence length.
- Each weight is a `mo.constant.external` with a name such as
  `gate_proj.weight`. The graph refers to weights by name and gets their
  values when it is loaded, so the same graph works with any checkpoint.
- On a GPU, each weight also passes through a `mo.transfer` that copies it
  from host memory to the device (yellow in the diagram); on a CPU there is
  nothing to copy.
- Every line is one operation (`rmo.matmul`, `mo.silu`, ...) with fully known
  types. That is what lets the compiler fuse operations, plan memory for the
  whole computation, and generate code for the target device ahead of time.

## Build a graph yourself

`max.graph` builds graphs directly. A `Graph` takes a name, the input types,
and a `forward` function that combines `max.graph.ops` operations. Weights
are declared as `Weight` objects with a name, dtype, shape, and device:

```python
H, I = config.hidden_size, config.intermediate_size


def mlp_forward(x):
    def weight(name: str, shape: list[int]) -> Weight:
        return Weight(
            name, DType.float32, shape, device=DeviceRef.from_device(DEVICE)
        )

    gate = x @ weight("gate_proj.weight", [I, H]).T
    up = x @ weight("up_proj.weight", [I, H]).T
    return (ops.silu(gate) * up) @ weight("down_proj.weight", [H, I]).T


graph = Graph(
    "mlp",
    forward=mlp_forward,
    input_types=[
        TensorType(
            DType.float32, ("seq_len", H), device=DeviceRef.from_device(DEVICE)
        )
    ],
)
```

An `InferenceSession` compiles graphs for a set of devices. `load` compiles
this one and binds each named weight from a registry; `execute` runs it on
`Buffer`s, the driver's plain device memory:

```python
session = InferenceSession(devices=[DEVICE])
compiled_mlp = session.load(
    graph, weights_registry=select(weights, "model.layers.0.mlp.")
)

x = np.random.default_rng(0).standard_normal((5, H)).astype(np.float32)
(out,) = compiled_mlp.execute(Buffer.from_numpy(x).to(DEVICE))
expected = mlp(Tensor(x, device=DEVICE)).to_numpy()
scale = np.abs(expected).max()
relative = np.abs(out.to_numpy() - expected).max() / scale
print(
    f"max difference from your module, relative to the largest output: {relative:.1e}"
)
assert relative < 1e-5
```

```text
max difference from your module, relative to the largest output: 4.8e-07
```

The hand-built graph and your module agree because `Module.compile` builds
this same graph for you, loads it in a session shared by the whole process,
and wraps the result so it takes and returns `Tensor`s.

Custom ops work in graphs too, through `ops.custom`. Compile the kernel from
[Section 3.1](03_01_custom_ops.md) again and use it in
place of `silu * up`:

```python
import mojo.notebook
```

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

```python
def fused_forward(x):
    def weight(name: str, shape: list[int]) -> Weight:
        return Weight(
            name, DType.float32, shape, device=DeviceRef.from_device(DEVICE)
        )

    gate = x @ weight("gate_proj.weight", [I, H]).T
    up = x @ weight("up_proj.weight", [I, H]).T
    (h,) = ops.custom(
        "swiglu",
        device=DeviceRef.from_device(DEVICE),
        values=[gate, up],
        out_types=[gate.type],
    )
    return h.tensor @ weight("down_proj.weight", [H, I]).T


fused_graph = Graph(
    "fused_mlp",
    forward=fused_forward,
    input_types=[
        TensorType(
            DType.float32, ("seq_len", H), device=DeviceRef.from_device(DEVICE)
        )
    ],
    custom_extensions=[Path("kernels.mojoc")],
)
fused_mlp = session.load(
    fused_graph, weights_registry=select(weights, "model.layers.0.mlp.")
)
(fused_out,) = fused_mlp.execute(Buffer.from_numpy(x).to(DEVICE))
relative = np.abs(fused_out.to_numpy() - expected).max() / scale
print(f"max difference, relative to the largest output: {relative:.1e}")
assert relative < 1e-5
```

```text
max difference, relative to the largest output: 4.8e-07
```

## Production weight loading

The loader from
[Section 1.3](01_03_tokens_and_weights.md) reads one
safetensors file into memory as `float32`. MAX's own loader,
`max.graph.weights.load_weights`, reads safetensors (including checkpoints split
across many files) and GGUF. It maps files into memory instead of reading them,
so a tensor is read only when it's used, and it keeps each tensor's original
dtype and, for GGUF, its quantization encoding. Tensors are reached by attribute
path:

```python
path = Path(hf_hub_download(MODEL_ID, "model.safetensors"))
checkpoint = load_weights([path])
gate = checkpoint.model.layers[0].mlp.gate_proj.weight
print(gate.name)
weight = gate.allocate()
print(weight.shape, weight.dtype)
```

```text
model.layers.0.mlp.gate_proj.weight
[Dim(1536), Dim(576)] DType.bfloat16
```

`allocate` returns a `Weight` like the ones in your hand-built graph, with
the checkpoint's name, shape, and dtype filled in. A serving pipeline builds
its graph from these, so the graph matches the checkpoint exactly, and the
loader supplies the data when the graph is loaded.

## Why it matters

Because the whole computation is known before it runs, MAX can:

- fuse operations and plan memory across the entire model instead of one
  operation at a time;
- compile once, ahead of time, and cache the result (the reason `max serve`
  starts faster the second time);
- on GPUs, record a decode step's kernel launches and replay them with almost
  no CPU overhead (*device graph capture*);
- run the same graph on CPUs and on NVIDIA, AMD, and Apple GPUs.

`max.nn` modules with `compile` give you all of this while you write ordinary
Python. Use `max.graph` directly when you need control over exactly what goes
into a graph, as MAX's serving pipelines do. MAX's docs describe the
[graph API](https://max.modular.com/develop/graph) in full; the handbook's
[kernel optimization](https://handbook.modular.com/kernel-optimization/kernel-optimization-for-llm-inference/)
page explains why whole-graph compilation and fusion matter for LLMs.

## Exercises

1. Print the full traced graph of one `DecoderLayer` from
   [Section 1.6](01_06_full_model.md). Find the flash
   attention call and the residual additions.
2. Build the MLP graph with a static input shape, `[5, 576]`, instead of
   `seq_len`. What happens when you execute it with 6 positions?
3. Load layer 0's MLP weights through `load_weights` instead of
   `load_checkpoint`. What has to change for the `float32` graph?

## Next

[Chapter 4](ch04.md) turns to correctness and speed:
testing layers against references, verifying a whole model, and benchmarking and
profiling.

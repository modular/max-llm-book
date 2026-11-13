# 1.4 Building blocks

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/modular/max-llm-book/blob/main/notebooks/01_04_building_blocks.ipynb)
[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/modular/max-llm-book/blob/main/notebooks/01_04_building_blocks.ipynb)
[![View on GitHub](https://img.shields.io/badge/GitHub-notebook-black?logo=github)](https://github.com/modular/max-llm-book/blob/main/notebooks/01_04_building_blocks.ipynb)

Each of SmolLM2's 30 layers has the same structure:

```text
x = x + attention(rms_norm(x))     # mix information across positions
x = x + mlp(rms_norm(x))           # transform each position on its own
```

This section builds the three pieces besides attention: RMSNorm, the SwiGLU
MLP, and rotary position embeddings (RoPE), which attention uses to know
where each token is. Each piece is a few lines with `max.nn` and
`max.functional`, and each is checked against a NumPy version of its
formula, using the real weights of layer 0.

```python
import logging

import numpy as np
import plotly.graph_objects as go
from max import functional as F
from max.driver import CPU
from max.dtype import DType
from max.nn import Linear, Module, RMSNorm
from max.tensor import Tensor, default_device, default_dtype

from minillm import viz
from minillm.device import pick_device
from minillm.config import LlamaConfig
from minillm.weights import load_checkpoint

logging.getLogger("max").setLevel(logging.ERROR)
DEVICE = pick_device()  # a GPU if MAX finds one, otherwise the CPU

config = LlamaConfig.from_pretrained()
weights = load_checkpoint()
```

A layer's weights are the checkpoint entries under its name prefix. This
helper selects them and strips the prefix, so a module can load them with
`load_state_dict`:

```python
def select(
    weights: dict[str, np.ndarray], prefix: str
) -> dict[str, np.ndarray]:
    """Returns the entries under ``prefix``, with the prefix removed."""
    return {
        name.removeprefix(prefix): value
        for name, value in weights.items()
        if name.startswith(prefix)
    }
```

<small>Saved to `minillm/weights.py`.</small>

```python
list(select(weights, "model.layers.0.mlp."))
```

```text
['down_proj.weight', 'gate_proj.weight', 'up_proj.weight']
```

A random input stands in for the hidden states that reach layer 0. `to_np`
reads a tensor back, and `compare` reports the largest difference from the
NumPy result, absolute and relative to the largest value:

```python
rng = np.random.default_rng(0)
x_np = rng.standard_normal((1, 6, config.hidden_size)).astype(np.float32)
x = Tensor(x_np, device=DEVICE)


def to_np(t: Tensor) -> np.ndarray:
    return t.to_numpy()


def compare(actual: np.ndarray, expected: np.ndarray) -> None:
    diff = np.abs(actual - expected).max()
    relative = diff / np.abs(expected).max()
    print(f"max difference {diff:.1e} (relative {relative:.1e})")
    assert relative < 1e-4, "doesn't match the NumPy reference"
```

## RMSNorm

Deep networks train better when each layer sees inputs of a stable scale.
RMSNorm divides each vector by its root mean square, then multiplies by a
learned weight per dimension:

\\[\\mathrm{RMSNorm}(x) = \\frac{x}{\\sqrt{\\mathrm{mean}(x^2) + \\epsilon}} \\cdot w\\]

Unlike LayerNorm, it doesn't subtract the mean or add a bias, which
saves work and in practice trains as well. `max.nn.RMSNorm` implements it:

```python
with default_device(DEVICE), default_dtype(DType.float32):
    norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
norm.load_state_dict(select(weights, "model.layers.0.input_layernorm."))

w = to_np(norm.weight)
expected = (
    x_np / np.sqrt((x_np**2).mean(-1, keepdims=True) + config.rms_norm_eps) * w
)
compare(to_np(norm(x)), expected)
```

```text
max difference 3.0e-08 (relative 7.7e-08)
```

The learned weight is far from the all-ones it starts as. In layer 0, most
dimensions are scaled close to zero, a few stand out several times larger, and
nearly half are negative, so the norm also flips their sign. Hover over a bar to
read its value:

```python
fig = go.Figure(
    go.Bar(
        x=np.arange(config.hidden_size),
        y=w,
        marker_color=viz.BLUE,
        hovertemplate="dimension %{x}: %{y:.3f}<extra></extra>",
    )
)
fig.update_layout(
    title="Layer 0 input RMSNorm weight, per dimension",
    height=340,
    bargap=0,
    xaxis_title="dimension",
    yaxis_title="weight",
)
viz.show(fig)
```

<div class="plotly-figure" data-src="generated/01_04_building_blocks_11_0.json"></div>

## The SwiGLU MLP

The MLP transforms each position independently. SmolLM2 uses a *gated*
variant: one projection, passed through SiLU, decides how much of a second
projection to let through, and a third projection maps back to the hidden
size:

\\[\\mathrm{MLP}(x) = W\_{\\text{down}}\\,\\big(\\mathrm{SiLU}(W\_{\\text{gate}}\\,x) \\odot W\_{\\text{up}}\\,x\\big)\\]

where \\(\\mathrm{SiLU}(z) = z \\cdot \\sigma(z)\\). The attribute names match the
checkpoint (`gate_proj`, `up_proj`, `down_proj`), so `load_state_dict` needs
no renaming.

```python
z = np.linspace(-6, 6, 241)
fig = go.Figure(
    [
        go.Scatter(
            x=z,
            y=np.maximum(z, 0),
            name="ReLU",
            line=dict(color=viz.ORANGE, dash="dot"),
        ),
        go.Scatter(
            x=z, y=z / (1 + np.exp(-z)), name="SiLU", line_color=viz.BLUE
        ),
    ]
)
fig.update_traces(hovertemplate="%{x:.2f} -> %{y:.3f}")
fig.update_layout(
    title="SiLU is a smooth ReLU that lets small negative values through",
    height=360,
    xaxis_title="z",
    yaxis_title="activation",
    hovermode="x unified",
)
viz.show(fig)
```

<div class="plotly-figure" data-src="generated/01_04_building_blocks_13_0.json"></div>

```python
from __future__ import annotations

from max import functional as F
from max.driver import Device
from max.dtype import DType
from max.graph import Dim
from max.nn import Linear, Module
from max.tensor import Tensor
from max.tensor.sharding import DeviceMesh

from minillm.config import LlamaConfig


class MLP(Module):
    """The SwiGLU feed-forward network of a Llama layer."""

    def __init__(self, config: LlamaConfig) -> None:
        hidden, inner = config.hidden_size, config.intermediate_size
        self.gate_proj = Linear(hidden, inner, bias=False)
        self.up_proj = Linear(hidden, inner, bias=False)
        self.down_proj = Linear(inner, hidden, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
```

<small>Saved to `minillm/layers.py`.</small>

```python
with F.lazy(), default_device(DEVICE), default_dtype(DType.float32):
    mlp = MLP(config)
mlp.load_state_dict(select(weights, "model.layers.0.mlp."))

gate, up, down = (
    to_np(getattr(mlp, n).weight) for n in ("gate_proj", "up_proj", "down_proj")
)
g = x_np @ gate.T
expected = (g / (1 + np.exp(-g)) * (x_np @ up.T)) @ down.T
compare(to_np(mlp(x)), expected)
```

```text
max difference 1.0e-03 (relative 8.2e-07)
```

`Linear` stores its weight as `[out_features, in_features]`, the same layout
as PyTorch, so the NumPy version multiplies by the transpose. The
differences, about one part in a million, come from summing thousands of
`float32` products in a different order than NumPy does.

## Rotary position embeddings

Attention compares tokens with dot products, which by themselves ignore
order: "dog bites man" and "man bites dog" contain the same tokens. The model
needs to know positions.

RoPE encodes position by *rotating* query and key vectors. It splits each
64-dimension head vector into 32 pairs and rotates pair *i* at position *p*
by the angle \\(p \\cdot \\theta\_i\\), where \\(\\theta\_i = \\text{base}^{-2i/d}\\).
Early pairs rotate quickly and late pairs slowly, like the hands of a clock.
SmolLM2's base is `rope_theta = 100000`.

Following the Hugging Face layout, the two halves of the vector form the
pairs: element *i* pairs with element *i + 32*. Rotating every pair at once
is then two multiplications:

\\[\\mathrm{rope}(x) = x \\cdot \\cos + \\mathrm{rotate\\\_half}(x) \\cdot \\sin,
\\qquad \\mathrm{rotate\\\_half}([x\_1, x\_2]) = [-x\_2, x\_1]\\]

`start` offsets the positions;
[Section 1.8](01_08_kv_cache.md) needs it for the KV
cache. The tables name their dtype: a `Tensor` built without one takes the
device's default, which is `bfloat16` on a GPU, and `bfloat16` can't even
represent every integer position above 256 exactly. Lengths are typed
`int | Dim` because a compiled model sees symbolic dimensions such as `seq_len`
instead of numbers;
[Section 1.2](01_02_tensors_and_eager.md#modules-and-compilation)
shows where they come from.

```python
def rope_cos_sin(
    head_dim: int,
    theta: float,
    start: int | Dim | Tensor,
    length: int | Dim,
    device: Device | DeviceMesh,
) -> tuple[Tensor, Tensor]:
    """Returns the RoPE cos and sin tables, ``[length, head_dim]``, for
    positions ``start`` to ``start + length - 1``."""
    inv_freq = Tensor(
        [theta ** (-i / head_dim) for i in range(0, head_dim, 2)],
        dtype=DType.float32,
        device=device,
    )
    positions = (
        Tensor.arange(length, dtype=DType.float32, device=device) + start
    )
    angles = F.outer(positions, inv_freq)
    angles = F.concat([angles, angles], axis=-1)
    return F.cos(angles), F.sin(angles)


def rotate_half(x: Tensor) -> Tensor:
    half = x.shape[-1] // 2
    return F.concat([-x[..., half:], x[..., :half]], axis=-1)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Rotates ``x``, shaped ``[batch, seq, heads, head_dim]``."""
    seq, dim = cos.shape
    cos = cos.reshape([1, seq, 1, dim])
    sin = sin.reshape([1, seq, 1, dim])
    return x * cos + rotate_half(x) * sin
```

<small>Saved to `minillm/layers.py`.</small>

Each line is one pair's rotation at a position; move the slider to
advance the position. Pair 0 turns a full radian per token,
while pair 31 has barely moved after 24 tokens:

```python
positions = 25
cos_t, sin_t = rope_cos_sin(
    config.head_dim, config.rope_theta, 0, positions, DEVICE
)
angles = np.arctan2(
    to_np(sin_t), to_np(cos_t)
)  # [position, head_dim], one angle per pair
pairs = [0, 2, 8, 31]
steps = []
for pos in range(positions):
    group = []
    for pair, color in zip(pairs, viz.SERIES):
        degrees = float(np.degrees(angles[pos, pair]))
        group.append(
            go.Scatterpolar(
                r=[0, 1],
                theta=[0, degrees],
                mode="lines+markers",
                name=f"pair {pair}",
                line=dict(color=color, width=3),
                marker=dict(size=[0, 10], color=color),
                hovertemplate=f"pair {pair} at position {pos}: {degrees:.0f} degrees<extra></extra>",
            )
        )
    steps.append(group)
fig = go.Figure(
    layout=dict(
        title="RoPE rotates each pair at its own speed",
        height=520,
        polar=dict(
            radialaxis=dict(visible=False, range=[0, 1.05]),
            angularaxis=dict(gridcolor=viz.GRID, linecolor=viz.GRID),
        ),
    )
)
viz.show(
    viz.with_slider(
        fig, steps, [str(p) for p in range(positions)], "position ", active=3
    )
)
```

<div class="plotly-figure" data-src="generated/01_04_building_blocks_19_0.json"></div>

RoPE relies on two properties, and both are easy to check. First, a
rotation doesn't change a vector's length:

```python
q = Tensor(
    rng.standard_normal((1, 8, 1, config.head_dim)).astype(np.float32),
    device=DEVICE,
)
cos, sin = rope_cos_sin(config.head_dim, config.rope_theta, 0, 8, DEVICE)
rotated = to_np(apply_rope(q, cos, sin))
print(np.linalg.norm(to_np(q), axis=-1).ravel()[:4])
print(np.linalg.norm(rotated, axis=-1).ravel()[:4])
assert np.allclose(
    np.linalg.norm(to_np(q), axis=-1),
    np.linalg.norm(rotated, axis=-1),
    rtol=1e-5,
)
```

```text
[6.897215  6.9866242 7.8358827 6.884518 ]
[6.897215  6.9866247 7.8358827 6.884518 ]
```

Second, the dot product of a rotated query and key depends only on how far
apart they are. Here a query at position 7 and a key at
position 3 score the same as at positions 107 and 103:

```python
def score(q_pos: int, k_pos: int) -> float:
    qv = Tensor(np.ones((1, 1, 1, config.head_dim), np.float32), device=DEVICE)
    kv = Tensor(
        np.linspace(-1, 1, config.head_dim, dtype=np.float32).reshape(
            1, 1, 1, -1
        ),
        device=DEVICE,
    )
    q_rot = apply_rope(
        qv, *rope_cos_sin(config.head_dim, config.rope_theta, q_pos, 1, DEVICE)
    )
    k_rot = apply_rope(
        kv, *rope_cos_sin(config.head_dim, config.rope_theta, k_pos, 1, DEVICE)
    )
    return float((to_np(q_rot) * to_np(k_rot)).sum())


print(f"{score(7, 3):.4f}  {score(107, 103):.4f}  {score(7, 5):.4f}")
assert (
    abs(score(7, 3) - score(107, 103)) < 1e-3
    and abs(score(7, 3) - score(7, 5)) > 1e-2
)
```

```text
10.2645  10.2645  7.8206
```

The first two scores match. The third, for positions two apart instead of
four, differs. The sweep below fixes the key at position 10 or at 110 and
moves the query up to 48 positions after it. The two curves lie on top of
each other:

```python
q_vec = np.ones(config.head_dim, np.float32)
k_vec = np.linspace(-1, 1, config.head_dim, dtype=np.float32)
cos_t, sin_t = (
    to_np(t)
    for t in rope_cos_sin(config.head_dim, config.rope_theta, 0, 160, DEVICE)
)


def rope_np(v: np.ndarray, pos: int) -> np.ndarray:
    half = len(v) // 2
    return v * cos_t[pos] + np.concatenate([-v[half:], v[:half]]) * sin_t[pos]


distances = np.arange(49)
fig = go.Figure(
    [
        go.Scatter(
            x=distances,
            y=[
                rope_np(q_vec, k_pos + d) @ rope_np(k_vec, k_pos)
                for d in distances
            ],
            name=f"key at position {k_pos}",
            line=dict(color=color, dash=dash, width=width),
            hovertemplate="distance %{x}: %{y:.3f}<extra></extra>",
        )
        for k_pos, color, dash, width in [
            (10, viz.BLUE, "solid", 5),
            (110, viz.ORANGE, "dash", 2),
        ]
    ]
)
fig.update_layout(
    title="Query-key score after RoPE depends only on the distance",
    height=380,
    xaxis_title="query position minus key position",
    yaxis_title="q · k",
)
viz.show(fig)
```

<div class="plotly-figure" data-src="generated/01_04_building_blocks_25_0.json"></div>

Because scores depend only on relative position, RoPE generalizes well,
and most open models use it.

## Exercises

1. Replace `RMSNorm` with a version you write from `F.rsqrt`, `.mean`, and
   `*`, and check that it gives the same output.
2. Plot `cos` from `rope_cos_sin` for 512 positions as an image (positions
   by dimensions). Which dimensions change fastest?
3. Run the RMSNorm check with `bfloat16` instead of `float32` (cast the input
   and the module's weight). How large is the error, and why do real models
   compute norms in `float32`?

## Next

[Section 1.5](01_05_attention.md) builds attention,
the part of the layer that mixes information between positions, and uses RoPE to
do it.

# 1.6 The full model

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/modular/max-llm-book/blob/main/notebooks/01_06_full_model.ipynb)
[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/modular/max-llm-book/blob/main/notebooks/01_06_full_model.ipynb)
[![View on GitHub](https://img.shields.io/badge/GitHub-notebook-black?logo=github)](https://github.com/modular/max-llm-book/blob/main/notebooks/01_06_full_model.ipynb)

You now have every part of SmolLM2. This section assembles them into a
decoder layer, stacks 30 layers into the model, adds the output head, loads
all 272 weights, and compiles the result. Then it checks that your model's
logits match MAX's own implementation of SmolLM2, the one `AutoModel`
loads.

```python
import logging
import time
import warnings

import numpy as np
import plotly.graph_objects as go
from max import functional as F
from max.driver import CPU
from max.dtype import DType
from max.nn import Embedding, Linear, Module, ModuleList, RMSNorm
from max.pipelines import AutoModel, AutoTokenizer
from max.tensor import Tensor, default_device, default_dtype
from max.tensor.tensor import TensorType

from minillm import viz
from minillm.attention import Attention
from minillm.device import pick_device
from minillm.config import MODEL_ID, LlamaConfig
from minillm.layers import MLP
from minillm.weights import load_checkpoint, select

logging.getLogger("max").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="The eager interpreter failed")
DEVICE = pick_device()  # a GPU if MAX finds one, otherwise the CPU

config = LlamaConfig.from_pretrained()
weights = load_checkpoint()
```

## The decoder layer

A layer normalizes its input, runs attention, and adds the result back to
the input; then it does the same with the MLP:

\\[\\begin{aligned}
h &= x + \\mathrm{Attention}(\\mathrm{RMSNorm}(x)) \\\\
x' &= h + \\mathrm{MLP}(\\mathrm{RMSNorm}(h))
\\end{aligned}\\]

Adding back (the *residual connection*) means each layer only refines the
hidden state instead of replacing it, which helps a stack of 30 layers
train well.

The attribute names match the checkpoint once more, so a layer loads
`model.layers.N.*` directly.

```python
from __future__ import annotations

from max import functional as F
from max.driver import Device
from max.dtype import DType
from max.nn import Embedding, Linear, Module, ModuleList, RMSNorm
from max.tensor import Tensor, default_device, default_dtype
from numpy.typing import NDArray

from minillm.attention import Attention
from minillm.config import LlamaConfig
from minillm.layers import MLP


class DecoderLayer(Module):
    """One transformer layer: attention and an MLP, each with a residual."""

    def __init__(self, config: LlamaConfig) -> None:
        eps = config.rms_norm_eps
        self.input_layernorm = RMSNorm(config.hidden_size, eps=eps)
        self.self_attn = Attention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=eps)
        self.mlp = MLP(config)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.self_attn(self.input_layernorm(x))
        return x + self.mlp(self.post_attention_layernorm(x))
```

<small>Saved to `minillm/model.py`.</small>

Run layer 0 on real token embeddings and compare the size of each token's
input with the size of the update the layer adds to it:

```python
tokenizer = AutoTokenizer(MODEL_ID)
ids = tokenizer(["The capital of France is"])["input_ids"][0]

with F.lazy(), default_device(DEVICE), default_dtype(DType.float32):
    layer = DecoderLayer(config)
layer.load_state_dict(select(weights, "model.layers.0."))

x = Tensor(weights["model.embed_tokens.weight"][ids][None], device=DEVICE)
y = layer(x)
x_np, y_np = x.to_numpy(), y.to_numpy()
print("input norm per token: ", np.linalg.norm(x_np, axis=-1).round(2))
print("update norm per token:", np.linalg.norm(y_np - x_np, axis=-1).round(2))
```

```text
input norm per token:  [[2.   2.54 2.91 2.28 2.6 ]]
update norm per token: [[47.39 51.99 31.03 68.12 35.3 ]]
```

The update is 10 to 30 times larger than the input, because embeddings
start as small vectors. The residual sum still carries the original
embedding forward. Later in this section, a plot follows this *residual
stream* through all 30 layers.

## The model

The model embeds the tokens, runs the 30 layers in order, and normalizes the
result. The causal LM wrapper adds `lm_head`, a `Linear` from 576 dimensions
to one logit per vocabulary entry. The module tree mirrors Hugging Face's
`LlamaForCausalLM`, so every checkpoint name is also a parameter name.

`load_model` builds the model inside `F.lazy()`, so the random
initialization never runs, and then loads the weights. SmolLM2 ties
`lm_head` to the embedding table, so the loader uses the same array for
both.

```python
class LlamaModel(Module):
    """The embedding, the decoder layers, and the final norm."""

    def __init__(self, config: LlamaConfig) -> None:
        self.embed_tokens = Embedding(config.vocab_size, dim=config.hidden_size)
        self.layers = ModuleList(
            [DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: Tensor) -> Tensor:
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class LlamaForCausalLM(Module):
    """A Llama model with its language-model head: token IDs in, logits out."""

    def __init__(self, config: LlamaConfig) -> None:
        self.model = LlamaModel(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids: Tensor) -> Tensor:
        return self.lm_head(self.model(input_ids))


def load_model(
    config: LlamaConfig,
    weights: dict[str, NDArray],
    device: Device,
    model_class: type[Module] = LlamaForCausalLM,
) -> Module:
    """Builds ``model_class`` on ``device`` and loads ``weights`` into it."""
    with F.lazy(), default_device(device), default_dtype(DType.float32):
        model = model_class(config)
    if config.tie_word_embeddings:
        weights = {
            **weights,
            "lm_head.weight": weights["model.embed_tokens.weight"],
        }
    model.load_state_dict(weights)
    return model
```

<small>Saved to `minillm/model.py`.</small>

Here is the whole model as a diagram, built from the config. The shaded
block repeats 30 times, and the curved lines are the residual connections.
Hover over a box for the shape of the tensor it produces:

```python
H, V, I = config.hidden_size, config.vocab_size, config.intermediate_size
stages = [  # (label, kind, output shape)
    ("token IDs", "io", "[batch, seq]"),
    (f"Embedding: {V:,} × {H}", "embedding", f"[batch, seq, {H}]"),
    ("RMSNorm", "norm", f"[batch, seq, {H}]"),
    (
        f"Attention: {config.num_attention_heads} query heads, {config.num_key_value_heads} key/value heads, RoPE",
        "attention",
        f"[batch, seq, {H}]",
    ),
    ("+", "add", f"[batch, seq, {H}]"),
    ("RMSNorm", "norm", f"[batch, seq, {H}]"),
    (f"SwiGLU MLP: {H} → {I} → {H}", "mlp", f"[batch, seq, {H}]"),
    ("+", "add", f"[batch, seq, {H}]"),
    ("final RMSNorm", "norm", f"[batch, seq, {H}]"),
    (
        f"LM head: {H} → {V:,} (shares the embedding)",
        "embedding",
        f"[batch, seq, {V}]",
    ),
    ("logits", "io", f"[batch, seq, {V}]"),
]
colors = {
    "io": viz.NEUTRAL,
    "embedding": viz.ORANGE,
    "norm": viz.VIOLET,
    "attention": viz.BLUE,
    "mlp": viz.AQUA,
    "add": viz.SURFACE,
}
ys = [-i for i in range(len(stages))]

fig = go.Figure()
fig.add_shape(
    type="rect",
    x0=-2.6,
    x1=2.6,
    y0=ys[7] - 0.45,
    y1=ys[2] + 0.75,
    line=dict(color=viz.GRID),
    fillcolor=viz.NEUTRAL,
    layer="below",
)
fig.add_annotation(
    x=2.5,
    y=ys[2] + 0.55,
    text=f"× {config.num_hidden_layers} layers",
    showarrow=False,
    xanchor="right",
    font=dict(color=viz.INK_SECONDARY),
)
for (label, kind, _), y in zip(stages, ys):
    width = 0.25 if kind == "add" else 2.1
    fig.add_shape(
        type="rect",
        x0=-width,
        x1=width,
        y0=y - 0.3,
        y1=y + 0.3,
        fillcolor=colors[kind],
        line=dict(
            color=viz.INK_SECONDARY if kind in ("add", "io") else colors[kind]
        ),
        layer="above" if kind == "add" else "below",
    )
    fig.add_annotation(
        x=0,
        y=y,
        text=label,
        showarrow=False,
        font=dict(
            color=viz.INK if kind in ("io", "add") else viz.SURFACE, size=13
        ),
    )
for upper, lower in zip(ys, ys[1:]):  # the main path
    fig.add_annotation(
        x=0,
        y=lower + 0.3,
        ax=0,
        ay=upper - 0.3,
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        showarrow=True,
        arrowhead=2,
        arrowcolor=viz.INK_SECONDARY,
        text="",
    )
for start, add in [
    (1, 4),
    (4, 7),
]:  # residual connections skip the norm and the sublayer
    y0, y1 = ys[start] - 0.5, ys[add]
    fig.add_trace(
        go.Scatter(
            x=[0, -2.35, -2.35, -0.25],
            y=[y0, y0, y1, y1],
            mode="lines",
            hoverinfo="skip",
            line=dict(color=viz.INK_SECONDARY, dash="dot"),
            showlegend=False,
        )
    )
fig.add_trace(
    go.Scatter(
        x=[0] * len(stages),
        y=ys,
        mode="markers",
        marker=dict(size=40, opacity=0),
        customdata=[[label, shape] for label, _, shape in stages],
        showlegend=False,
        hovertemplate="%{customdata[0]}<br>output: %{customdata[1]}<extra></extra>",
    )
)
fig.update_layout(
    title="SmolLM2-135M, from token IDs to logits",
    height=640,
    xaxis=dict(visible=False, range=[-2.8, 2.8]),
    yaxis=dict(visible=False, range=[ys[-1] - 0.6, 0.6]),
    margin=dict(l=8, r=8),
)
viz.show(fig)
```

<div class="plotly-figure" data-src="generated/01_06_full_model_11_0.json"></div>

```python
model = load_model(config, weights, DEVICE)
n_params = sum(p.num_elements() for _, p in model.parameters)
print(
    f"{len(list(model.parameters))} parameter tensors, {n_params / 1e6:.1f}M values"
)
```

```text
273 parameter tensors, 162.8M values
```

Before compiling, run the layers one at a time and keep each one's output.
The size of each token's hidden state grows as layers add their updates:

```python
x = model.model.embed_tokens(Tensor(np.array([ids]), device=DEVICE))
states = [x.to_numpy()[0]]
for layer in model.model.layers:
    x = layer(x)
    states.append(x.to_numpy()[0])
states = np.array(states)  # [layer + 1, token, hidden]
labels = [repr(tokenizer.decode([i])) for i in ids]

fig = go.Figure(
    [
        go.Scatter(
            x=np.arange(31),
            y=np.linalg.norm(states[:, t], axis=-1),
            name=label,
            line_color=viz.SERIES[t],
            mode="lines",
            hovertemplate=label + " after layer %{x}: %{y:.1f}<extra></extra>",
        )
        for t, label in enumerate(labels)
    ]
)
fig.update_layout(
    title="Hidden-state norm through the layers (0 is the embedding)",
    height=420,
    xaxis_title="layer",
    yaxis=dict(title="norm (log scale)", type="log", dtick=1),
)
viz.show(fig)
```

<div class="plotly-figure" data-src="generated/01_06_full_model_14_0.json"></div>

The first token behaves differently. Its state jumps to about 24,000 at layer
12, over a hundred times the other tokens' (about 120 to 165), and stays there
until it drops in the last two layers. Models use the first position as a
resting place for attention
([Section 1.5](01_05_attention.md)), and this large
state is part of how they do it.

Every layer writes into the same stream, so you can ask what the model would
predict if it stopped early: apply the final norm and `lm_head` to each
layer's output. For the hidden state \\(h\_\\ell\\) after layer \\(\\ell\\), the
*logit lens* reads

\\[p\_\\ell = \\mathrm{softmax}\\big(E \\, \\mathrm{RMSNorm}(h\_\\ell)\\big)\\]

where \\(E\\) is the embedding table that `lm_head` shares. Applied to the
last position, it shows the prediction forming. Hover to see each layer's top
token:

```python
final_norm = weights["model.norm.weight"]
table = weights["model.embed_tokens.weight"]  # lm_head shares it
paris = tokenizer([" Paris"])["input_ids"][0][0]
last = states[:, -1]
normed = (
    last
    / np.sqrt((last**2).mean(-1, keepdims=True) + config.rms_norm_eps)
    * final_norm
)
lens = normed @ table.T
lens_probs = np.exp(lens - lens.max(-1, keepdims=True))
lens_probs /= lens_probs.sum(-1, keepdims=True)
top_tokens = [repr(tokenizer.decode([int(i)])) for i in lens.argmax(-1)]

fig = go.Figure(
    go.Scatter(
        x=np.arange(31),
        y=lens_probs[:, paris],
        mode="lines+markers",
        line_color=viz.BLUE,
        customdata=top_tokens,
        name="P(' Paris')",
        hovertemplate="layer %{x}: P(' Paris') = %{y:.2f}<br>top token %{customdata}<extra></extra>",
    )
)
fig.update_layout(
    title="Logit lens: the probability of ' Paris' after each layer",
    height=400,
    xaxis_title="layer",
    yaxis=dict(title="probability", range=[0, 1]),
)
viz.show(fig)
```

<div class="plotly-figure" data-src="generated/01_06_full_model_16_0.json"></div>

Through most of the stack the top token is filler: a period or a space,
then " the" from layer 23. " Paris" becomes the top token only after layer
30, the last. Its probability stays near zero through layer 28, reaches
about 0.06 after layer 29 and about 0.47 after layer 30, so for this prompt
the answer forms in the last two layers.

Before these plots, the model printed 273 parameter tensors and 162.8M
values. That is 28M more than the checkpoint's 134.5M because `lm_head`
counts as its own 49152 × 576 parameter, loaded from the same array as the
embedding table. `load_state_dict` is strict: a
missing or misnamed weight raises an error instead of silently keeping a
random value, so a successful load already proves the names line up.

## Compile

Running 30 layers eagerly works but executes hundreds of operations one at
a time. Compiling turns the whole forward pass into one optimized graph.
The input is a batch of token-ID sequences of any length:

```python
start = time.perf_counter()
compiled = model.compile(
    TensorType(DType.int64, ("batch", "seq_len"), device=DEVICE)
)
print(f"compiled in {time.perf_counter() - start:.0f} s")

start = time.perf_counter()
logits = compiled(Tensor(np.array([ids]), device=DEVICE)).to_numpy()
print(
    f"forward pass: {(time.perf_counter() - start) * 1000:.0f} ms, logits {logits.shape}"
)
```

```text
compiled in 10 s
forward pass: 30 ms, logits (1, 5, 49152)
```

## Check against MAX's SmolLM2

`AutoModel` loads SmolLM2 with MAX's own implementation of the architecture
that `config.json` names `LlamaForCausalLM`, and can return the logits for
each token it generates.
The first generated token's logits must match your model's logits at the
last prompt position:

```python
reference_model = AutoModel(MODEL_ID, quantization_encoding="float32")
reference = reference_model.generate(
    input_ids=[ids], max_new_tokens=1, output_logits=True
)
expected = reference.logits[0][0]
reference_model.close()

ours = logits[0, -1]
print(
    f"max difference {np.abs(ours - expected).max():.1e} (logits range up to {np.abs(expected).max():.0f})"
)
print("top token, yours:", repr(tokenizer.decode([int(ours.argmax())])))
print("top token, MAX's:", repr(tokenizer.decode([int(expected.argmax())])))
assert (
    ours.argmax() == expected.argmax() and np.abs(ours - expected).max() < 1e-2
)
```

```text
max difference 4.0e-05 (logits range up to 21)
top token, yours: ' Paris'
top token, MAX's: ' Paris'
```

Plot your logits against MAX's for the 2,000 likeliest tokens. The points sit
on the diagonal:

```python
likeliest = np.argsort(expected)[::-1][:2000]
words = [repr(tokenizer.decode([int(i)])) for i in likeliest]
fig = go.Figure(
    go.Scatter(
        x=expected[likeliest],
        y=ours[likeliest],
        mode="markers",
        text=words,
        marker=dict(color=viz.BLUE, size=6, opacity=0.6),
        hovertemplate="%{text}<br>MAX %{x:.4f}<br>yours %{y:.4f}<extra></extra>",
    )
)
fig.update_layout(
    title="Next-token logits: your model against MAX's",
    height=480,
    xaxis_title="MAX's logit",
    yaxis=dict(title="your logit", scaleanchor="x"),
)
viz.show(fig)
```

<div class="plotly-figure" data-src="generated/01_06_full_model_23_0.json"></div>

In the run shown here, on an Apple-silicon Mac with a Metal GPU, the largest
difference between your logits and MAX's was about `4e-5`, and both models
predict the same next token. Treat these numbers as an example: yours depend on
your hardware and vary from run to run. An error in any layer, such as a wrong
RoPE layout or a missing scale, would change the logits by whole units, so the
match shows that every layer you wrote is correct.

The same code runs the larger SmolLM2 models by changing `MODEL_ID`. Other
Llama-family models need small additions: Llama 3.1, for example, rescales
RoPE's frequencies (`rope_scaling` in its config), and Qwen2 adds biases to the
query, key, and value projections. The handbook's
[Choosing the right model](https://handbook.modular.com/getting-started/choosing-the-right-model/)
discusses how size, architecture, and license trade off when you pick one.

## Exercises

1. The logit lens above reads only the last position. Plot it for every
   position: at which layer does each one settle on its final prediction?
2. Load `HuggingFaceTB/SmolLM2-360M-Instruct` with the same code. What
   changes in `config`, and how do compile and forward times scale?
3. Break something on purpose, such as dropping `rotate_half` from
   `apply_rope`, and see how far the logits move.

## Next

[Section 1.7](01_07_sampling.md) turns logits into
text: the generation loop, and the sampling strategies behind `temperature`,
`top_k`, and `top_p`.

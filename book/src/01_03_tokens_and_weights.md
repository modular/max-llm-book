# 1.3 Tokens, embeddings, and weights

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/modular/max-llm-book/blob/main/notebooks/01_03_tokens_and_weights.ipynb)
[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/modular/max-llm-book/blob/main/notebooks/01_03_tokens_and_weights.ipynb)
[![View on GitHub](https://img.shields.io/badge/GitHub-notebook-black?logo=github)](https://github.com/modular/max-llm-book/blob/main/notebooks/01_03_tokens_and_weights.ipynb)

From this section on, you rebuild SmolLM2-135M-Instruct piece by piece. A
model on Hugging Face is two things: a configuration that gives its sizes,
and a checkpoint that holds its trained weights. This section loads both,
then builds the model's first layer, the token embedding.

As in [Section 1.2](01_02_tensors_and_eager.md), code
cells that start with `# export:` go into the `minillm` package.

```python
import json
import logging
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from max.driver import CPU
from max.dtype import DType
from max.nn import Embedding
from max.pipelines import AutoTokenizer
from max.tensor import Tensor, default_device, default_dtype

from minillm import viz
from minillm.device import pick_device

logging.getLogger("max").setLevel(logging.ERROR)
DEVICE = pick_device()  # a GPU if MAX finds one, otherwise the CPU
```

## The configuration

`config.json` lists the architecture and its sizes. These numbers decide
the shape of every weight:

```python
from huggingface_hub import hf_hub_download

MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
raw = json.loads(Path(hf_hub_download(MODEL_ID, "config.json")).read_text())
{
    k: raw[k]
    for k in [
        "architectures",
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "rope_theta",
        "rms_norm_eps",
        "tie_word_embeddings",
    ]
}
```

```text
{'architectures': ['LlamaForCausalLM'],
 'vocab_size': 49152,
 'hidden_size': 576,
 'intermediate_size': 1536,
 'num_hidden_layers': 30,
 'num_attention_heads': 9,
 'num_key_value_heads': 3,
 'rope_theta': 100000,
 'rms_norm_eps': 1e-05,
 'tie_word_embeddings': True}
```

Read it as: 30 identical layers (`num_hidden_layers`), each working on vectors
of 576 numbers (`hidden_size`). Attention uses 9 query heads but only 3
key/value heads, so each head has 576 / 9 = 64 dimensions;
[Section 1.5](01_05_attention.md) explains why the
counts differ. `LlamaForCausalLM` is the architecture the code must implement.

A small dataclass keeps these fields typed. The field names match Hugging
Face's, so you can compare against any Llama-family `config.json`.

<details class="code-fold">
<summary>Show <code>LlamaConfig</code> <span class="code-fold-lines">44 lines</span></summary>

```python
from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path

from huggingface_hub import hf_hub_download

MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"


@dataclass(frozen=True)
class LlamaConfig:
    """The sizes of a Llama-family model, as in its ``config.json``."""

    vocab_size: int = 49152
    hidden_size: int = 576
    intermediate_size: int = 1536
    num_hidden_layers: int = 30
    num_attention_heads: int = 9
    num_key_value_heads: int = 3
    rope_theta: float = 100000.0
    rms_norm_eps: float = 1e-5
    max_position_embeddings: int = 8192
    tie_word_embeddings: bool = True
    eos_token_id: int = 2
    head_dim: int = 0  # 0 means hidden_size // num_attention_heads

    def __post_init__(self) -> None:
        # Newer configs, such as Qwen3's, give head_dim explicitly.
        if not self.head_dim:
            object.__setattr__(
                self, "head_dim", self.hidden_size // self.num_attention_heads
            )

    @classmethod
    def from_pretrained(cls, model_id: str = MODEL_ID) -> LlamaConfig:
        """Reads the config of a Hugging Face model."""
        raw = json.loads(
            Path(hf_hub_download(model_id, "config.json")).read_text()
        )
        return cls(
            **{f.name: raw[f.name] for f in fields(cls) if f.name in raw}
        )
```

</details>

<small>Saved to `minillm/config.py`.</small>

```python
config = LlamaConfig.from_pretrained()
print(config)
print("head_dim:", config.head_dim)
```

```text
LlamaConfig(vocab_size=49152, hidden_size=576, intermediate_size=1536, num_hidden_layers=30, num_attention_heads=9, num_key_value_heads=3, rope_theta=100000, rms_norm_eps=1e-05, max_position_embeddings=8192, tie_word_embeddings=True, eos_token_id=2, head_dim=64)
head_dim: 64
```

## The checkpoint

The weights live in `model.safetensors`. The format is simple enough to read
by hand: 8 bytes giving the length of a JSON header, the header itself, then
the raw bytes of every tensor. The header maps each tensor's name to its
dtype, shape, and byte range:

```python
path = hf_hub_download(MODEL_ID, "model.safetensors")
with open(path, "rb") as f:
    header_len = int.from_bytes(f.read(8), "little")
    header = json.loads(f.read(header_len))
header.pop("__metadata__", None)
print(len(header), "tensors")
first = list(header)[:8]
width = max(len(name) for name in first)
for name in first:
    print(f"{name:{width}} {header[name]}")
```

```text
272 tensors
model.embed_tokens.weight                      {'dtype': 'BF16', 'shape': [49152, 576], 'data_offsets': [0, 56623104]}
model.layers.0.input_layernorm.weight          {'dtype': 'BF16', 'shape': [576], 'data_offsets': [56623104, 56624256]}
model.layers.0.mlp.down_proj.weight            {'dtype': 'BF16', 'shape': [576, 1536], 'data_offsets': [56624256, 58393728]}
model.layers.0.mlp.gate_proj.weight            {'dtype': 'BF16', 'shape': [1536, 576], 'data_offsets': [58393728, 60163200]}
model.layers.0.mlp.up_proj.weight              {'dtype': 'BF16', 'shape': [1536, 576], 'data_offsets': [60163200, 61932672]}
model.layers.0.post_attention_layernorm.weight {'dtype': 'BF16', 'shape': [576], 'data_offsets': [61932672, 61933824]}
model.layers.0.self_attn.k_proj.weight         {'dtype': 'BF16', 'shape': [192, 576], 'data_offsets': [61933824, 62155008]}
model.layers.0.self_attn.o_proj.weight         {'dtype': 'BF16', 'shape': [576, 576], 'data_offsets': [62155008, 62818560]}
```

The names follow the module tree of Hugging Face's `LlamaForCausalLM`: an
embedding, then `model.layers.0` through `model.layers.29`, each with
attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`), an MLP
(`gate_proj`, `up_proj`, `down_proj`), and two norms. Count the parameters:

```python
n_params = sum(int(np.prod(t["shape"])) for t in header.values())
print(f"{n_params / 1e6:.1f}M parameters")
print(
    f"bfloat16: {n_params * 2 / 2**20:.0f} MiB, float32: {n_params * 4 / 2**20:.0f} MiB"
)
print("lm_head.weight in checkpoint:", "lm_head.weight" in header)
```

```text
134.5M parameters
bfloat16: 257 MiB, float32: 513 MiB
lm_head.weight in checkpoint: False
```

Where do the parameters live? Click a box to zoom in, and click the bar at
the top to zoom back out:

<details class="code-fold">
<summary>Show <code>part</code>, <code>layer_number</code>, <code>order</code> <span class="code-fold-lines">71 lines</span></summary>

```python
def part(name: str) -> str:
    if "embed_tokens" in name:
        return "embedding"
    if "self_attn" in name:
        return "attention"
    if "mlp" in name:
        return "MLP"
    return "norms"


color_of = {
    "embedding": viz.ORANGE,
    "attention": viz.BLUE,
    "MLP": viz.AQUA,
    "norms": viz.VIOLET,
}
# id -> [parent, label, total, color]
nodes = {"SmolLM2-135M": ["", "SmolLM2-135M", 0, viz.NEUTRAL]}


def layer_number(name: str) -> int:
    return int(name.split(".")[2]) if ".layers." in name else -1


def order(name: str) -> int:
    """Sorts the embedding first, then the layers in order, then the norm."""
    if "embed_tokens" in name:
        return -1
    return layer_number(name) if ".layers." in name else 99


for name in sorted(header, key=order):
    count = int(np.prod(header[name]["shape"]))
    kind = part(name)
    layer = layer_number(name)
    path = (
        ["SmolLM2-135M"]
        + ([f"layer {layer}"] if layer >= 0 else [])
        + [kind, name]
    )
    for depth in range(1, len(path)):
        node = "/".join(path[: depth + 1])
        label = name.split(".")[-2] if depth == len(path) - 1 else path[depth]
        color = color_of[kind] if path[depth] in (kind, name) else viz.NEUTRAL
        nodes.setdefault(node, ["/".join(path[:depth]), label, 0, color])
        nodes[node][2] += count
    nodes["SmolLM2-135M"][2] += count

fig = go.Figure(
    go.Treemap(
        ids=list(nodes),
        parents=[n[0] for n in nodes.values()],
        labels=[n[1] for n in nodes.values()],
        values=[n[2] for n in nodes.values()],
        branchvalues="total",
        sort=False,
        maxdepth=3,
        marker=dict(
            colors=[n[3] for n in nodes.values()],
            line=dict(color=viz.SURFACE, width=2),
        ),
        hovertemplate="%{label}<br>%{value:,} parameters, %{percentRoot:.1%} of the model<extra></extra>",
        texttemplate="%{label}<br>%{value:.3s}",
    )
)
fig.update_layout(
    title="SmolLM2-135M's parameters",
    height=520,
    margin=dict(t=56, l=8, r=8, b=8),
)
viz.show(fig)
```

</details>

<div class="plotly-figure" data-src="generated/01_03_tokens_and_weights_13_0.json"></div>

- Memory is parameters times bytes per parameter. The checkpoint stores
  `bfloat16` (2 bytes); this book converts to `float32` (4 bytes), so its
  results can be compared exactly with reference implementations. The handbook's
  [GPU memory calculation](https://handbook.modular.com/getting-started/calculating-gpu-memory-for-llms/)
  applies the same arithmetic to large models, plus the KV cache you build in
  [Section 1.8](01_08_kv_cache.md).
- There is no `lm_head.weight`. With `tie_word_embeddings`, the output layer
  reuses the embedding matrix;
  [Section 1.6](01_06_full_model.md) uses that.

The loader below maps the file into memory and returns NumPy arrays. NumPy
has no `bfloat16` type, but the conversion is exact and cheap: a `bfloat16`
value is the upper 16 bits of a `float32`, so shifting the bits left by 16
gives the same number as a `float32`. MAX modules accept NumPy arrays
directly, so this is all the loading code the model needs.

<details class="code-fold">
<summary>Show <code>read_safetensors</code>, <code>load_checkpoint</code> <span class="code-fold-lines">44 lines</span></summary>

```python
from __future__ import annotations

import json

import numpy as np
from huggingface_hub import hf_hub_download

from minillm.config import MODEL_ID

_NUMPY_DTYPES = {
    "F32": np.float32,
    "F16": np.float16,
    "I64": np.int64,
    "I32": np.int32,
}


def read_safetensors(path: str) -> dict[str, np.ndarray]:
    """Reads a safetensors file, converting floating-point tensors to float32."""
    with open(path, "rb") as f:
        header_len = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(header_len))
    data = np.memmap(path, dtype=np.uint8, mode="r", offset=8 + header_len)
    tensors = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        start, end = info["data_offsets"]
        raw = data[start:end]
        if info["dtype"] == "BF16":
            array = (raw.view(np.uint16).astype(np.uint32) << 16).view(
                np.float32
            )
        else:
            array = raw.view(_NUMPY_DTYPES[info["dtype"]])
            if array.dtype == np.float16:
                array = array.astype(np.float32)
        tensors[name] = array.reshape(info["shape"])
    return tensors


def load_checkpoint(model_id: str = MODEL_ID) -> dict[str, np.ndarray]:
    """Downloads a model's safetensors checkpoint and returns its tensors."""
    return read_safetensors(hf_hub_download(model_id, "model.safetensors"))
```

</details>

<small>Saved to `minillm/weights.py`.</small>

```python
weights = load_checkpoint()
w = weights["model.embed_tokens.weight"]
print(w.shape, w.dtype)
```

```text
(49152, 576) float32
```

## Token embeddings

The model's first layer turns each token ID into a vector of 576 numbers.
It is a lookup table: row *i* of a `[49152, 576]` matrix is the vector for
token *i*. `max.nn.Embedding` implements it. `load_state_dict` replaces its
random weight with the checkpoint's:

```python
with default_device(DEVICE), default_dtype(DType.float32):
    embed = Embedding(config.vocab_size, dim=config.hidden_size)
embed.load_state_dict({"weight": weights["model.embed_tokens.weight"]})

tokenizer = AutoTokenizer(MODEL_ID)
ids = tokenizer([" Paris France banana"])["input_ids"][0]
vectors = embed(Tensor(np.array([ids]), device=DEVICE))
print(ids, [tokenizer.decode([i]) for i in ids])
print(vectors.shape)
```

```text
[7042, 4649, 21285] [' Paris', ' France', ' banana']
[Dim(1), Dim(3), Dim(576)]
```

Check it: each output vector must equal its row of the weight matrix.

```python
table = embed.weight.to_numpy()
v = vectors.to_numpy()[0]
assert np.array_equal(v, table[ids])
print("embedding rows match the checkpoint")
```

```text
embedding rows match the checkpoint
```

Training places related tokens near each other. Cosine similarity, the cosine
of the angle between two vectors, shows it even before any layer runs:

\\[\\cos\\theta = \\frac{a \\cdot b}{\\lVert a \\rVert \\, \\lVert b \\rVert}\\]

It is 1 for vectors pointing the same way, 0 for unrelated (orthogonal)
ones, and \\(-1\\) for opposite ones, whatever their lengths.

```python
def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


print(f"Paris  ~ France: {cosine(v[0], v[1]):.2f}")
print(f"Paris  ~ banana: {cosine(v[0], v[2]):.2f}")
print(f"France ~ banana: {cosine(v[1], v[2]):.2f}")
```

```text
Paris  ~ France: 0.59
Paris  ~ banana: 0.36
France ~ banana: 0.37
```

Three numbers don't show much. Project the embeddings of a few dozen words
from three groups onto the two directions along which they vary most
(principal component analysis, done here with NumPy's SVD). Hover over a
point for its word:

```python
groups = {
    "countries": "France Italy Germany Spain Japan China Brazil Canada Egypt India",
    "fruits": "apple banana orange grape mango lemon cherry peach pear melon",
    "numbers": "one two three four five six seven eight nine ten",
}
words, group_of, rows = [], [], []
for group, members in groups.items():
    for word in members.split():
        token = tokenizer([" " + word])["input_ids"][0]
        if len(token) == 1:  # keep words that are a single token
            words.append(word)
            group_of.append(group)
            rows.append(table[token[0]])
points = np.array(rows) - np.mean(rows, axis=0)
_, _, axes = np.linalg.svd(points, full_matrices=False)
xy = points @ axes[:2].T
```

<details class="code-fold">
<summary>Show the code <span class="code-fold-lines">30 lines</span></summary>

```python
fig = go.Figure()
for (group, color), symbol in zip(
    zip(groups, viz.SERIES), ["circle", "square", "diamond"]
):
    index = [i for i, g in enumerate(group_of) if g == group]
    fig.add_trace(
        go.Scatter(
            x=xy[index, 0],
            y=xy[index, 1],
            mode="markers+text",
            name=group,
            text=[words[i] for i in index],
            textposition="top center",
            textfont=dict(size=11),
            marker=dict(
                color=color,
                symbol=symbol,
                size=10,
                line=dict(color=viz.SURFACE, width=2),
            ),
            hovertemplate="%{text}<extra>" + group + "</extra>",
        )
    )
fig.update_layout(
    title="Word embeddings projected to two dimensions",
    height=520,
    xaxis=dict(title="first principal component", showticklabels=False),
    yaxis=dict(title="second principal component", showticklabels=False),
)
viz.show(fig)
```

</details>

<div class="plotly-figure" data-src="generated/01_03_tokens_and_weights_25_0.json"></div>

The groups separate without the model ever being told what a country or a
fruit is: words used in similar contexts during training end up with
similar vectors.

The layers refine these vectors using context: what `Paris` means depends
on the tokens around it.

## Exercises

1. Find the 5 tokens whose embeddings are most similar to `" Paris"`. (Hint:
   normalize the table's rows, then take one matrix-vector product.)
2. The treemap counts the embedding once because the output layer reuses
   it. How many parameters, and how many MiB in `float32`, would SmolLM2
   need with a separate `lm_head`?

## Next

[Section 1.4](01_04_building_blocks.md) builds the
layers that run inside each of the 30 blocks: RMSNorm, rotary position
embeddings, and the SwiGLU MLP.

# 1.8 The KV cache

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/modular/max-llm-book/blob/main/notebooks/01_08_kv_cache.ipynb)
[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/modular/max-llm-book/blob/main/notebooks/01_08_kv_cache.ipynb)
[![View on GitHub](https://img.shields.io/badge/GitHub-notebook-black?logo=github)](https://github.com/modular/max-llm-book/blob/main/notebooks/01_08_kv_cache.ipynb)

[Section 1.7](01_07_sampling.md) ended with a problem:
each generated token reruns the model over the whole sequence, so the cost per
token grows with the text. This section fixes it the way inference engines do,
with a *KV cache*, and measures the difference.

## Why caching works

To compute attention for a new token, the model needs the keys and values of
every earlier position, in every layer. Those keys and values depend only
on the tokens up to their own position, so they never change once computed.
Recomputing them every step is wasted work. The KV cache stores them
instead, which splits generation into two phases:

- *Prefill* runs the whole prompt at once, like before, and saves every
  layer's keys and values.
- *Decode* runs one new token per step. Each layer computes the new
  token's query, key, and value, appends the key and value to its cache, and
  attends over the whole cache.

The handbook's
[How does LLM inference work?](https://handbook.modular.com/llm-inference-basics/how-does-llm-inference-work/)
page describes the same two phases from the serving side.

<details class="code-fold">
<summary>Show the setup <span class="code-fold-lines">27 lines</span></summary>

```python
import logging
import time
import warnings

import numpy as np
import plotly.graph_objects as go
from max.driver import CPU
from plotly.subplots import make_subplots
from max.dtype import DType
from max.graph import TensorType
from max.pipelines import AutoTokenizer
from max.tensor import Tensor

from minillm import viz
from minillm.device import pick_device
from minillm.config import MODEL_ID, LlamaConfig
from minillm.model import load_model
from minillm.sampling import generate
from minillm.weights import load_checkpoint

logging.getLogger("max").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="The eager interpreter failed")
DEVICE = pick_device()  # a GPU if MAX finds one, otherwise the CPU

config = LlamaConfig.from_pretrained()
weights = load_checkpoint()
tokenizer = AutoTokenizer(MODEL_ID)
```

</details>

The difference in work is easy to see. Each row below is one generation
step for a 6-token prompt; each column is a position whose key and value
that step computes. Without a cache, every step recomputes every earlier
position. With one, prefill computes the prompt once and each decode step
computes one new position:

```python
prompt_len, steps = 6, 10
step_labels = ["prefill"] + [f"decode {i}" for i in range(1, steps)]
recompute_work = np.zeros((steps, prompt_len + steps - 1))
cached_work = np.zeros_like(recompute_work)
for step in range(steps):
    end = prompt_len + step
    recompute_work[step, :end] = 1
    cached_work[step, :end] = 1 if step == 0 else 0
    cached_work[step, end - 1] = 1
```

<details class="code-fold">
<summary>Show the code <span class="code-fold-lines">28 lines</span></summary>

```python
fig = make_subplots(
    rows=1,
    cols=2,
    shared_yaxes=True,
    horizontal_spacing=0.06,
    subplot_titles=[
        f"Recompute: {int(recompute_work.sum())} key/value computations",
        f"KV cache: {int(cached_work.sum())} key/value computations",
    ],
)
for col, work in enumerate([recompute_work, cached_work], start=1):
    fig.add_trace(
        go.Heatmap(
            z=np.where(work > 0, 1, np.nan),
            y=step_labels,
            xgap=2,
            ygap=2,
            colorscale=[[0, viz.BLUE], [1, viz.BLUE]],
            showscale=False,
            hovertemplate="%{y}: computes position %{x}<extra></extra>",
        ),
        row=1,
        col=col,
    )
fig.update_xaxes(title="position", showgrid=False, dtick=2)
fig.update_yaxes(autorange="reversed", showgrid=False)
fig.update_layout(height=380, title="Which keys and values each step computes")
viz.show(fig)
```

</details>

<div class="plotly-figure" data-src="generated/01_08_kv_cache_5_0.json"></div>

## How big is the cache?

Each token stores one key and one value vector per key/value head, per
layer. With \\(L\\) layers, \\(H\_{kv}\\) key/value heads of dimension \\(d\\), and \\(b\\)
bytes per number:

\\[\\text{bytes per token} = 2 \\cdot L \\cdot H\_{kv} \\cdot d \\cdot b
= 2 \\cdot 30 \\cdot 3 \\cdot 64 \\cdot 4\\]

```python
per_token = (
    2
    * config.num_hidden_layers
    * config.num_key_value_heads
    * config.head_dim
    * 4
)
print(f"{per_token / 1024:.0f} KiB per token in float32")
print(
    f"{per_token * 8192 / 2**20:.0f} MiB for SmolLM2's full 8192-token context"
)
print(
    f"{per_token * 3 * 8192 / 2**20:.0f} MiB if every query head had its own key/value head"
)
```

```text
45 KiB per token in float32
360 MiB for SmolLM2's full 8192-token context
1080 MiB if every query head had its own key/value head
```

The cache grows with the context and with the number of sequences served at
once. Compare it with the model's weights:

```python
weights_mib = sum(w.nbytes for w in weights.values()) / 2**20
two_sequences_mib = per_token * 8192 * 2 / 2**20
print(f"weights: {weights_mib:.0f} MiB")
print(f"cache for two sequences of 8192 tokens: {two_sequences_mib:.0f} MiB")
```

```text
weights: 513 MiB
cache for two sequences of 8192 tokens: 720 MiB
```

Move the slider to serve more sequences at once:

<details class="code-fold">
<summary>Show the code <span class="code-fold-lines">43 lines</span></summary>

```python
lengths = np.arange(256, 8193, 256)
batches = [1, 2, 4, 16, 64]
steps = []
for batch in batches:
    steps.append(
        [
            go.Scatter(
                x=lengths,
                y=per_token * lengths * batch / 2**20,
                name="KV cache, 3 key/value heads",
                line_color=viz.BLUE,
                hovertemplate="%{x} tokens: %{y:,.0f} MiB<extra></extra>",
            ),
            go.Scatter(
                x=lengths,
                y=3 * per_token * lengths * batch / 2**20,
                name="KV cache, 9 key/value heads",
                line=dict(color=viz.ORANGE, dash="dot"),
                hovertemplate="%{x} tokens: %{y:,.0f} MiB<extra></extra>",
            ),
        ]
    )
fig = go.Figure(
    go.Scatter(
        x=[lengths[0], lengths[-1]],
        y=[weights_mib] * 2,
        name="weights",
        mode="lines",
        line=dict(color=viz.INK_SECONDARY, dash="dash"),
        hovertemplate="weights: %{y:,.0f} MiB<extra></extra>",
    )
)
fig.update_layout(
    title="float32 KV cache memory against context length",
    height=460,
    xaxis_title="context length (tokens)",
    yaxis=dict(title="MiB", type="log", dtick=1),
)
viz.show(
    viz.with_slider(
        fig, steps, [str(b) for b in batches], "sequences served at once: "
    )
)
```

</details>

<div class="plotly-figure" data-src="generated/01_08_kv_cache_11_0.json"></div>

This is where grouped-query attention
([Section 1.5](01_05_attention.md)) pays off: 3
key/value heads instead of 9 make the cache 3 times smaller. Even for SmolLM2 in
`float32`, two sequences at the full 8,192-token context need a 720 MiB cache,
more than the 513 MiB of weights. For large models serving many users, the cache
often takes more GPU memory than the weights. The handbook's
[GPU memory calculation](https://handbook.modular.com/getting-started/calculating-gpu-memory-for-llms/)
adds it to the weights when sizing a deployment.

## Attention with a cache

`CachedAttention` extends
[Section 1.5](01_05_attention.md)'s `Attention`. Its `forward` takes the
layer's cached keys and values, if any, and returns them extended by the new
positions. `start` is the position of the first new token, so RoPE rotates
the new queries and keys for where they really are.

Prefill uses the causal mask as before. Decode has one query, the newest
token, which may attend to every cached position, so it needs no mask.

<details class="code-fold">
<summary>Show <code>CachedAttention</code> <span class="code-fold-lines">37 lines</span></summary>

```python
from __future__ import annotations

import math

import numpy as np
from max import functional as F
from max.driver import Device
from max.dtype import DType
from max.graph import TensorType
from max.nn import ModuleList
from max.nn.kernels import flash_attention
from max.tensor import Tensor

from minillm.attention import Attention
from minillm.config import LlamaConfig
from minillm.model import DecoderLayer, LlamaForCausalLM, LlamaModel

KVPair = tuple[Tensor, Tensor]


class CachedAttention(Attention):
    """Attention that reads and extends a key/value cache."""

    def forward(
        self, x: Tensor, start: Tensor, past: KVPair | None = None
    ) -> tuple[Tensor, KVPair]:
        q, k, v = self.project(x, start)
        if past is None:  # prefill: causal attention over the prompt
            return self.attend(q, k, v), (k, v)
        k = F.concat([past[0], k], axis=1)
        v = F.concat([past[1], v], axis=1)
        # The one new token may attend to every cached position: no mask.
        out = flash_attention(q, k, v, 1 / math.sqrt(self.head_dim))
        out = out.rebind_mapping(q.mapping)
        batch, seq = q.shape[0], q.shape[1]
        out = out.reshape([batch, seq, self.num_heads * self.head_dim])
        return self.o_proj(out), (k, v)
```

</details>

<small>Saved to `minillm/kv_cache.py`.</small>

The layer and model pass each layer's cache through. The model returns
logits for the last position only: generation never needs the others, and
skipping them saves most of the LM head's work during prefill.

<details class="code-fold">
<summary>Show <code>CachedDecoderLayer</code>, <code>CachedLlamaModel</code>, <code>CachedLlamaForCausalLM</code> <span class="code-fold-lines">52 lines</span></summary>

```python
class CachedDecoderLayer(DecoderLayer):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__(config)
        self.self_attn = CachedAttention(config)

    def forward(
        self, x: Tensor, start: Tensor, past: KVPair | None = None
    ) -> tuple[Tensor, KVPair]:
        attn, kv = self.self_attn(self.input_layernorm(x), start, past)
        x = x + attn
        return x + self.mlp(self.post_attention_layernorm(x)), kv


class CachedLlamaModel(LlamaModel):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__(config)
        self.layers = ModuleList(
            [
                CachedDecoderLayer(config)
                for _ in range(config.num_hidden_layers)
            ]
        )

    def forward(
        self,
        input_ids: Tensor,
        start: Tensor,
        cache: list[KVPair] | None = None,
    ) -> tuple[Tensor, list[KVPair]]:
        x = self.embed_tokens(input_ids)
        new_cache = []
        for i, layer in enumerate(self.layers):
            x, kv = layer(x, start, None if cache is None else cache[i])
            new_cache.append(kv)
        return self.norm(x), new_cache


class CachedLlamaForCausalLM(LlamaForCausalLM):
    """Returns the last position's logits and the updated cache."""

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__(config)
        self.model = CachedLlamaModel(config)

    def forward(
        self,
        input_ids: Tensor,
        start: Tensor,
        cache: list[KVPair] | None = None,
    ) -> tuple[Tensor, list[KVPair]]:
        hidden, new_cache = self.model(input_ids, start, cache)
        return self.lm_head(hidden[:, -1]), new_cache
```

</details>

<small>Saved to `minillm/kv_cache.py`.</small>

## Two compiled graphs

Prefill and decode have different inputs, so they compile into two graphs
that share the same weights. The cache is a list of 30 `(key, value)` pairs.
Its length, `cache_len`, is a named dimension, so one decode graph serves
every step. `start` is a one-element tensor rather than a Python number, so
that it is an input of the graph instead of a constant baked into it. It is
`float32` because RoPE computes its angles in `float32`, and `float32` holds
every position up to 2^24 exactly.

`KVCacheDecoder` keeps the cache between calls. It has the same interface as
[Section 1.7](01_07_sampling.md)'s `next_logits`: given the tokens so far, it returns the next
token's logits, running a prefill for a new prompt and a decode step when
exactly one token was added. It plugs straight into `generate`.

<details class="code-fold">
<summary>Show <code>KVCacheDecoder</code> <span class="code-fold-lines">41 lines</span></summary>

```python
class KVCacheDecoder:
    """Returns next-token logits like a full forward pass, using a cache."""

    def __init__(
        self, model: CachedLlamaForCausalLM, config: LlamaConfig, device: Device
    ) -> None:
        self.device = device
        tokens = TensorType(DType.int64, ("batch", "seq_len"), device=device)
        one_token = TensorType(DType.int64, ("batch", 1), device=device)
        start = TensorType(DType.float32, [1], device=device)
        kv = TensorType(
            DType.float32,
            ("batch", "cache_len", config.num_key_value_heads, config.head_dim),
            device=device,
        )
        self.prefill = model.compile(tokens, start)
        self.decode = model.compile(
            one_token, start, [(kv, kv)] * config.num_hidden_layers
        )
        self.cache: list[KVPair] | None = None
        self.length = 0

    def __call__(self, ids: list[int]) -> Tensor:
        if self.cache is not None and len(ids) == self.length + 1:
            logits, self.cache = self.decode(
                self._tokens(ids[-1:]), self._position(self.length), self.cache
            )
        else:
            logits, self.cache = self.prefill(
                self._tokens(ids), self._position(0)
            )
        self.length = len(ids)
        return logits[0]

    def _tokens(self, ids: list[int]) -> Tensor:
        return Tensor(np.array([ids], dtype=np.int64), device=self.device)

    def _position(self, position: int) -> Tensor:
        return Tensor(
            np.array([position], dtype=np.float32), device=self.device
        )
```

</details>

<small>Saved to `minillm/kv_cache.py`.</small>

```python
start = time.perf_counter()
cached = KVCacheDecoder(
    load_model(config, weights, DEVICE, CachedLlamaForCausalLM), config, DEVICE
)
print(f"compiled prefill and decode in {time.perf_counter() - start:.0f} s")
```

```text
compiled prefill and decode in 30 s
```

## Check it

The cache must not change the result. Generate greedily with
[Section 1.7](01_07_sampling.md)'s full recompute and with the cache, and compare the tokens:

```python
full = load_model(config, weights, DEVICE).compile(
    TensorType(DType.int64, ("batch", "seq_len"), device=DEVICE)
)


def recompute(ids: list[int]) -> Tensor:
    return full(Tensor(np.array([ids]), device=DEVICE))[0, -1]


prompt = tokenizer(["The capital of France is"])["input_ids"][0]
slow = generate(recompute, prompt, 48, temperature=0)
fast = generate(cached, prompt, 48, temperature=0)
print("identical tokens:", slow == fast)
assert slow == fast
print(tokenizer.decode(prompt + fast))
print("cached keys per layer:", [int(d) for d in cached.cache[0][0].shape])
```

```text
identical tokens: True
The capital of France is Paris. Paris is the largest city in France and the capital of the French department of the Espace de Paris.

The Eiffel Tower, also known as the Eiffel Tower, is a monumental structure that stands 324
cached keys per layer: [1, 52, 3, 64]
```

## Measure it

Time one decode step at several context lengths, with random prompts so that
only the length matters:

```python
def ms_per_token(next_logits, prompt: list[int], steps: int = 24) -> float:
    ids = list(prompt)
    ids.append(int(np.argmax(next_logits(ids).to_numpy())))  # prefill, untimed
    start = time.perf_counter()
    for _ in range(steps):
        ids.append(int(np.argmax(next_logits(ids).to_numpy())))
    return (time.perf_counter() - start) / steps * 1000
```

```python
rng = np.random.default_rng(0)
lengths = [32, 128, 256, 512, 1024]
timings = {"recompute": [], "KV cache": []}
for length in lengths:
    prompt = rng.integers(0, config.vocab_size, length).tolist()
    timings["recompute"].append(ms_per_token(recompute, prompt))
    timings["KV cache"].append(ms_per_token(cached, prompt))
for name, ms in timings.items():
    print(f"{name:10}", "  ".join(f"{t:6.1f}" for t in ms), "ms")
```

```text
recompute   100.9   130.0   134.0   209.5   595.8 ms
KV cache    112.9   136.1   110.8   135.5   169.9 ms
```

<details class="code-fold">
<summary>Show the code <span class="code-fold-lines">21 lines</span></summary>

```python
fig = go.Figure(
    [
        go.Scatter(
            x=lengths,
            y=ms,
            name=name,
            mode="lines+markers",
            line_color=color,
            hovertemplate=name + ", %{x} tokens: %{y:.1f} ms<extra></extra>",
        )
        for (name, ms), color in zip(timings.items(), [viz.ORANGE, viz.BLUE])
    ]
)
fig.update_layout(
    title="Time per decode step",
    height=420,
    xaxis_title="context length (tokens)",
    yaxis_title="ms per token",
    yaxis_rangemode="tozero",
)
viz.show(fig)
```

</details>

<div class="plotly-figure" data-src="generated/01_08_kv_cache_24_0.json"></div>

Hover over the points to read the timings; yours depend on your hardware and
vary from run to run. Recompute grows with the context, because every step
reruns attention over the whole sequence. The cached step grows much more
slowly. Below a few hundred tokens the cache doesn't pay off, and can even be
slower, because fixed work, such as launching kernels and reading every
weight, dominates a step of a model this small.

MAX's own decode step is far faster: the TPOT measured in [Section
1.1](01_01_run_and_serve.md) was about 10 ms on the
same machine. Part of the gap is this section's simple design: every decode call
passes the 60 cache tensors in and gets 60 new ones back, `F.concat` copies each
layer's whole cache into a new tensor, and the loop reads the logits back to
NumPy to pick each token on the host. MAX's engine writes new keys and values
into preallocated cache blocks that its attention kernel reads in place, and it
samples on the GPU. [Section
5.3](05_03_paged_kv_cache.md) builds a block-based
cache and explains how production engines avoid the copy.

This simple cache has limits that production engines address:

- Memory is allocated per sequence, as it grows. A server doesn't know how
  long each reply will be, and contiguous per-sequence buffers fragment
  memory. *Paged attention* stores the cache in fixed-size blocks instead,
  like virtual memory pages.
- It serves one sequence at a time. A decode step for one token barely uses
  the hardware. Engines batch many sequences per step, which needs a cache
  that holds sequences of different lengths.
- Shared prompts are recomputed. Requests that start with the same system
  prompt could share those cache blocks (*prefix caching*).

[Chapter 5](ch05.md) builds these into a small serving
engine: [Section 5.2](05_02_continuous_batching.md)
adds continuous batching,
[Section 5.3](05_03_paged_kv_cache.md) a paged KV
cache, and [Section 5.4](05_04_prefix_caching.md)
prefix caching. The handbook covers each one:
[PagedAttention](https://handbook.modular.com/inference-optimization/pagedattention/),
[batching](https://handbook.modular.com/inference-optimization/static-dynamic-continuous-batching/),
and
[prefix caching](https://handbook.modular.com/inference-optimization/prefix-caching/).

## Exercises

1. Extend the timing chart to context lengths from 16 to 2048. Below what
   length does the cache stop paying off on your device, and why?
2. Make `KVCacheDecoder` keep the cache across two turns of a conversation:
   prefill only the new turn's tokens after the first reply.
3. Store the cache in `bfloat16` instead of `float32`. How much memory does
   it save, and do the greedy tokens change?

## What you built

Chapter 1 is complete. You have a working LLM inference stack written with MAX:
a tokenizer and weight loader, a Llama-architecture model that matches MAX's
own, sampling, and a KV cache. [Chapter 2](ch02.md)
splits this model across several devices with MAX's distributed tensors.

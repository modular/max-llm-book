# 5.1 The life of a request

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/modular/max-llm-book/blob/main/notebooks/05_01_request_lifecycle.ipynb)
[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/modular/max-llm-book/blob/main/notebooks/05_01_request_lifecycle.ipynb)
[![View on GitHub](https://img.shields.io/badge/GitHub-notebook-black?logo=github)](https://github.com/modular/max-llm-book/blob/main/notebooks/05_01_request_lifecycle.ipynb)

So far your model served one prompt at a time, from a notebook. A serving engine
takes requests from many users at once and keeps them all moving: it admits new
requests, runs the model, hands each request its next token, and stops each one
when it is done. [This chapter](ch05.md) builds that
engine over five sections, on top of a new model interface designed for it.

MAX Serve and other inference engines share this structure:

```text
requests ─► scheduler ─► model step (one batched forward pass) ─► sampler
               ▲                                                     │
               └─────────── KV cache manager ◄──── new tokens ◄──────┘
```

The handbook's
[How does LLM inference work?](https://handbook.modular.com/llm-inference-basics/how-does-llm-inference-work/)
page describes the same loop.

```python
import logging
import math
import time
import warnings
from dataclasses import dataclass, field

import numpy as np
import plotly.graph_objects as go
from max import functional as F
from max.driver import CPU
from max.dtype import DType
from max.nn import ModuleList
from max.nn.kernels.attention import flash_attention
from max.pipelines import AutoTokenizer
from max.tensor import Tensor
from max.tensor.tensor import TensorType
from plotly.subplots import make_subplots

from minillm import viz
from minillm.attention import Attention
from minillm.device import pick_device
from minillm.config import MODEL_ID, LlamaConfig
from minillm.layers import rotate_half
from minillm.model import DecoderLayer, LlamaForCausalLM, load_model
from minillm.weights import load_checkpoint

logging.getLogger("max").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="The eager interpreter failed")
DEVICE = pick_device()  # a GPU if MAX finds one, otherwise the CPU

config = LlamaConfig.from_pretrained()
tokenizer = AutoTokenizer(MODEL_ID)
```

## A model step for an engine

The cached model from [Section 1.8](01_08_kv_cache.md)
kept its cache inside and handled one sequence. An engine needs one call that
advances *several* sequences, each with its own history, by any number of
tokens. The *step* model does that. Each call takes, for every sequence in the
batch:

- the new tokens and their positions,
- an attention mask saying which past and new positions each token may see,
- the past keys and values,

and returns the next-token logits plus only the *new* keys and values. The model
keeps no state. Where the cache lives and how it is laid out is up to the
engine, and [Section 5.3](05_03_paged_kv_cache.md) and
[Section 5.4](05_04_prefix_caching.md) change exactly
that.

Positions now differ per sequence, so
[RoPE](01_04_building_blocks.md#rotary-position-embeddings)
takes a position for each token instead of a start offset:

```python
from __future__ import annotations

import math
from collections.abc import Callable

from max import functional as F
from max.dtype import DType
from max.nn import ModuleList
from max.nn.kernels.attention import flash_attention
from max.tensor import Tensor

from minillm.attention import Attention
from minillm.config import LlamaConfig
from minillm.layers import rotate_half
from minillm.model import DecoderLayer, LlamaForCausalLM

KVPair = tuple[Tensor, Tensor]
Step = Callable[..., tuple[Tensor, list[KVPair]]]
# Masked scores use a large finite value instead of -inf: attention kernels
# process keys in blocks, and a row whose first block is fully -inf computes
# exp(-inf - -inf) = NaN, which then spreads through the values to other rows.
MASKED = -1e30


def rope_at(
    x: Tensor, positions: Tensor, head_dim: int, theta: float
) -> Tensor:
    """Rotates ``x``, ``[batch, q, heads, head_dim]``, for per-token
    ``positions``, ``[batch, q]``."""
    inv_freq = Tensor(
        [theta ** (-i / head_dim) for i in range(0, head_dim, 2)],
        dtype=DType.float32,
        device=x.mesh,
    )
    batch, q = positions.shape
    angles = positions.reshape([batch, q, 1]) * inv_freq
    angles = F.concat([angles, angles], axis=-1).reshape(
        [batch, q, 1, head_dim]
    )
    return x * F.cos(angles) + rotate_half(x) * F.sin(angles)


class StepAttention(Attention):
    """Attends from new tokens to the past keys and values plus their own."""

    def forward(
        self, x: Tensor, positions: Tensor, mask: Tensor, past: KVPair
    ) -> tuple[Tensor, KVPair]:
        batch, q, _ = x.shape
        query = self.q_proj(x).reshape(
            [batch, q, self.num_heads, self.head_dim]
        )
        key = self.k_proj(x).reshape(
            [batch, q, self.num_kv_heads, self.head_dim]
        )
        value = self.v_proj(x).reshape(
            [batch, q, self.num_kv_heads, self.head_dim]
        )
        query = rope_at(query, positions, self.head_dim, self.rope_theta)
        key = rope_at(key, positions, self.head_dim, self.rope_theta)
        all_keys = F.concat([past[0], key], axis=1)
        all_values = F.concat([past[1], value], axis=1)
        # The mask has one column per past and new position.
        mask = F.rebind(mask, [batch, q, all_keys.shape[1]])
        out = flash_attention(
            query, all_keys, all_values, 1 / math.sqrt(self.head_dim), mask
        ).rebind_mapping(query.mapping)
        out = out.reshape([batch, q, self.num_heads * self.head_dim])
        return self.o_proj(out), (key, value)


class StepLayer(DecoderLayer):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__(config)
        self.self_attn = StepAttention(config)

    def forward(
        self, x: Tensor, positions: Tensor, mask: Tensor, past: KVPair
    ) -> tuple[Tensor, KVPair]:
        attn, kv = self.self_attn(
            self.input_layernorm(x), positions, mask, past
        )
        x = x + attn
        return x + self.mlp(self.post_attention_layernorm(x)), kv


class StepModel(LlamaForCausalLM):
    """Advances a batch of sequences: returns each one's next-token logits
    and the new tokens' keys and values for every layer."""

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__(config)
        self.model.layers = ModuleList(
            [StepLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        tokens: Tensor,
        positions: Tensor,
        mask: Tensor,
        past: list[KVPair],
    ) -> tuple[Tensor, list[KVPair]]:
        x = self.model.embed_tokens(tokens)
        new_kv = []
        for layer, layer_past in zip(self.model.layers, past, strict=True):
            x, kv = layer(x, positions, mask, layer_past)
            new_kv.append(kv)
        return self.lm_head(self.model.norm(x)[:, -1]), new_kv
```

<small>Saved to `minillm/step.py`.</small>

Compile it once. Every dimension that changes between calls is symbolic: the
batch size, the number of new tokens `q`, and the past length:

```python
import numpy as np
from max.driver import Device
from max.tensor.tensor import TensorType


def compile_step(model: StepModel, config: LlamaConfig, device: Device) -> Step:
    """Compiles ``model`` for any batch size, number of new tokens, and past
    length."""
    kv = TensorType(
        DType.float32,
        ("batch", "past", config.num_key_value_heads, config.head_dim),
        device=device,
    )
    return model.compile(
        TensorType(DType.int64, ("batch", "q"), device=device),
        TensorType(DType.float32, ("batch", "q"), device=device),
        TensorType(DType.float32, ("batch", "q", "kv"), device=device),
        [(kv, kv)] * config.num_hidden_layers,
    )
```

<small>Saved to `minillm/step.py`.</small>

```python
start = time.perf_counter()
weights = load_checkpoint()
step = compile_step(
    load_model(config, weights, DEVICE, StepModel), config, DEVICE
)
print(f"compiled the step model in {time.perf_counter() - start:.0f} s")
```

```text
compiled the step model in 12 s
```

## The cache and the batch

The engine stores each sequence's keys and values in a cache object. The
first version is the simplest: one growing array per layer. Every later
cache has the same interface (`length`, `num_layers`, `read`, and `append`),
so `run_step` and the step model work with all of them unchanged.

```python
class ContiguousCache:
    """One sequence's keys and values: a growing array per layer."""

    def __init__(self, config: LlamaConfig) -> None:
        shape = (0, config.num_key_value_heads, config.head_dim)
        self.layers = [
            [np.zeros(shape, np.float32), np.zeros(shape, np.float32)]
            for _ in range(config.num_hidden_layers)
        ]

    @property
    def length(self) -> int:
        return len(self.layers[0][0])

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    def read(self, layer: int) -> tuple[np.ndarray, np.ndarray]:
        return self.layers[layer][0], self.layers[layer][1]

    def append(self, layer: int, keys: np.ndarray, values: np.ndarray) -> None:
        entry = self.layers[layer]
        entry[0] = np.concatenate([entry[0], keys])
        entry[1] = np.concatenate([entry[1], values])
```

<small>Saved to `minillm/step.py`.</small>

To run several sequences in one step, the engine packs them into rectangular
arrays. Sequences add different numbers of new tokens, so shorter ones are
*left-padded*, and the mask is built per token: a new token may see its own
sequence's past and the new tokens up to itself, and nothing else. A padding
token sees only itself, so its row of the softmax stays well defined; its
output is discarded. Masked positions get a large negative number instead of
\\(-\\infty\\). If a padding row's first block of keys were entirely \\(-\\infty\\), a
blockwise attention kernel would compute \\(e^{-\\infty + \\infty}\\), which is
NaN. In the next layer that NaN would become the padding position's keys and
values, and the real tokens of the same sequence would pick it up even
though the mask hides that position.

```python
def run_step(
    step: Step, caches: list, new_tokens: list[list[int]], device: Device
) -> np.ndarray:
    """Advances each sequence by its ``new_tokens``, appends their keys and
    values to its cache, and returns next-token logits, ``[batch, vocab]``."""
    batch, q = len(caches), max(len(t) for t in new_tokens)
    past = max(c.length for c in caches)
    tokens = np.zeros((batch, q), np.int64)
    positions = np.zeros((batch, q), np.float32)
    mask = np.full((batch, q, past + q), MASKED, np.float32)
    for i, (cache, new) in enumerate(zip(caches, new_tokens, strict=True)):
        pad, n = q - len(new), cache.length
        tokens[i, pad:] = new
        positions[i, pad:] = np.arange(n, n + len(new))
        mask[i, pad:, :n] = 0.0  # every new token sees the sequence's past
        mask[i, pad:, past + pad :] = np.triu(
            np.full((len(new), len(new)), MASKED), 1
        )
        mask[i, np.arange(pad), past + np.arange(pad)] = (
            0.0  # padding sees itself
        )
    past_kv = []
    for layer in range(caches[0].num_layers):
        keys = np.zeros(
            (batch, past, *caches[0].read(layer)[0].shape[1:]), np.float32
        )
        values = np.zeros_like(keys)
        for i, cache in enumerate(caches):
            k, v = cache.read(layer)
            keys[i, : len(k)], values[i, : len(v)] = k, v
        past_kv.append(
            (Tensor(keys, device=device), Tensor(values, device=device))
        )
    logits, new_kv = step(
        Tensor(tokens, device=device),
        Tensor(positions, device=device),
        Tensor(mask, device=device),
        past_kv,
    )
    for layer, (k, v) in enumerate(new_kv):
        k, v = k.to_numpy(), v.to_numpy()
        for i, (cache, new) in enumerate(zip(caches, new_tokens, strict=True)):
            pad = q - len(new)
            cache.append(layer, k[i, pad:], v[i, pad:])
    return logits.to_numpy()
```

<small>Saved to `minillm/step.py`.</small>

Check the step against the model from
[Section 1.6](01_06_full_model.md) by generating
greedily for two prompts of different lengths in the *same* steps:

```python
full = load_model(config, weights, DEVICE).compile(
    TensorType(DType.int64, ("batch", "seq"), device=DEVICE)
)


def greedy_reference(ids: list[int], n: int) -> list[int]:
    ids = list(ids)
    for _ in range(n):
        ids.append(
            int(
                full(Tensor(np.array([ids]), device=DEVICE))
                .to_numpy()[0, -1]
                .argmax()
            )
        )
    return ids[-n:]


prompts = [
    tokenizer([t])["input_ids"][0]
    for t in ["The capital of France is", "Once upon a time, there was a"]
]
caches = [ContiguousCache(config) for _ in prompts]
outputs = [[int(t)] for t in run_step(step, caches, prompts, DEVICE).argmax(-1)]
for _ in range(11):
    next_ids = run_step(
        step, caches, [[o[-1]] for o in outputs], DEVICE
    ).argmax(-1)
    for out, t in zip(outputs, next_ids):
        out.append(int(t))
for prompt, out in zip(prompts, outputs):
    print(out == greedy_reference(prompt, 12), repr(tokenizer.decode(out)))
    assert out == greedy_reference(prompt, 12)
```

```text
True ' Paris. Paris is the largest city in France and the capital'
True ' young woman named Sarah who was struggling to find her place in'
```

Both sequences match the reference, even though they have different lengths
and shared every step.

Here is the mask `run_step` builds for a mixed step: sequence 0 has 6
tokens cached and decodes 1 new token, while sequence 1 has nothing cached
and prefills a 4-token prompt. A wrapper around `step` records the inputs it
receives. Blue cells may attend; hover over any cell:

```python
recorded = {}


def recording_step(tokens, positions, mask, past):
    recorded.update(
        tokens=tokens.to_numpy(),
        positions=positions.to_numpy(),
        mask=mask.to_numpy(),
    )
    return step(tokens, positions, mask, past)


caches = [ContiguousCache(config), ContiguousCache(config)]
run_step(
    step, caches[:1], [prompts[1][:6]], DEVICE
)  # sequence 0 now has 6 cached tokens
run_step(recording_step, caches, [[prompts[1][6]], prompts[0][:4]], DEVICE)

mask, positions = recorded["mask"], recorded["positions"]
past = mask.shape[2] - mask.shape[1]
columns = [f"cached slot {j}" for j in range(past)] + [
    f"new slot {j}" for j in range(mask.shape[1])
]
fig = make_subplots(
    rows=1,
    cols=2,
    horizontal_spacing=0.26,
    subplot_titles=[
        "sequence 0: 6 cached, decodes 1",
        "sequence 1: nothing cached, prefills 4",
    ],
)
for i, new_length in enumerate([1, 4]):
    pad = mask.shape[1] - new_length  # left padding, as in run_step
    rows = [
        f"padding row {r}"
        if r < pad
        else f"new token at position {int(positions[i, r])}"
        for r in range(mask.shape[1])
    ]
    text = [
        [
            f"{rows[r]} -> {columns[c]}: "
            + ("visible" if mask[i, r, c] == 0 else "masked")
            for c in range(mask.shape[2])
        ]
        for r in range(mask.shape[1])
    ]
    fig.add_trace(
        go.Heatmap(
            z=(mask[i] == 0).astype(int),
            zmin=0,
            zmax=1,
            showscale=False,
            xgap=2,
            ygap=2,
            colorscale=[
                [0, viz.GRID],
                [0.5, viz.GRID],
                [0.5, viz.BLUE],
                [1, viz.BLUE],
            ],
            x=columns,
            y=rows,
            customdata=text,
            hovertemplate="%{customdata}<extra></extra>",
        ),
        row=1,
        col=i + 1,
    )
fig.update_xaxes(showgrid=False, tickangle=-45)
fig.update_yaxes(showgrid=False, autorange="reversed")
fig.update_layout(
    title="One step's attention mask for a batch of two sequences",
    height=440,
    margin=dict(l=8),
)
viz.show(fig)
```

<div class="plotly-figure" data-src="generated/05_01_request_lifecycle_15_0.json"></div>

Sequence 0's single new token sits in the last row, after three padding rows
that each see only themselves. It sees its 6 cached slots and itself.
Sequence 1 has no cached tokens, so its 6 cached slots are masked in every
row. They hold zeros, there only because the batch's longest past is 6. Its
4 new tokens form the triangle of the
[causal mask](01_05_attention.md#the-causal-mask).

## A first engine: one request at a time

A request carries its prompt, how many tokens it may generate, and timestamps
for the latency metrics from
[Section 1.1](01_01_run_and_serve.md#measure-it), such
as time to first token (TTFT). The simplest engine serves requests in arrival
order, each to completion, one at a time:

```python
from dataclasses import dataclass, field


@dataclass
class Request:
    prompt: list[int]
    max_new_tokens: int
    output: list[int] = field(default_factory=list)
    arrived: float = 0.0
    first_token: float | None = None
    finished: float | None = None

    @property
    def done(self) -> bool:
        return len(self.output) >= self.max_new_tokens
```

<small>Saved to `minillm/step.py`.</small>

```python
def serve_one_at_a_time(requests: list[Request]) -> None:
    for request in requests:
        cache = ContiguousCache(config)
        logits = run_step(step, [cache], [request.prompt], DEVICE)
        while True:
            request.output.append(int(logits[0].argmax()))
            request.first_token = request.first_token or time.perf_counter()
            if request.done:
                break
            logits = run_step(step, [cache], [[request.output[-1]]], DEVICE)
        request.finished = time.perf_counter()


def make_workload() -> list[Request]:
    texts = [
        "Tell me about Paris.",
        "Write a haiku about the sea.",
        "What is 2 + 2?",
        "List three colors.",
        "Explain gravity in one sentence.",
        "Name a fruit.",
        "Describe a cat.",
        "Say hello in French.",
    ]
    now = time.perf_counter()
    return [
        Request(tokenizer([t])["input_ids"][0], max_new_tokens=n, arrived=now)
        for t, n in zip(texts, [32, 24, 8, 16, 32, 8, 24, 16])
    ]


def report(requests: list[Request]) -> None:
    ttft = [(r.first_token - r.arrived) * 1000 for r in requests]
    total = max(r.finished for r in requests) - min(r.arrived for r in requests)
    tokens = sum(len(r.output) for r in requests)
    print(
        f"TTFT: first request {ttft[0]:.0f} ms, median {np.median(ttft):.0f} ms, worst {max(ttft):.0f} ms"
    )
    print(f"{tokens} tokens in {total:.1f} s = {tokens / total:.0f} tokens/s")


sequential = make_workload()
serve_one_at_a_time(sequential)
report(sequential)
```

```text
TTFT: first request 317 ms, median 25305 ms, worst 50618 ms
160 tokens in 56.2 s = 3 tokens/s
```

Eight users sent their requests at the same moment. The first one got a token
almost at once. The last one waited for every request ahead of it to finish, so
its time to first token is most of the total run time.
[Section 4.3](04_03_kernel_benchmarks.md#benchmark-a-layer)
measured a decoder layer's decode step for 32 sequences at about 1.2 to 1.3
times the cost of a step for one, on the Mac the book was built on, so this
engine leaves most of the hardware's capacity unused.
[Section 5.2](05_02_continuous_batching.md) runs
several requests in each step.

The handbook's
[LLM inference metrics](https://handbook.modular.com/llm-inference-basics/llm-inference-metrics/)
page explains why serving systems track TTFT percentiles as well as averages.

## Exercises

1. Serve the workload in order of `max_new_tokens`, shortest first. What
   happens to the median and worst TTFT? What would a user with a long
   request think of this policy?
2. Run `run_step` with three prompts in one call and check each against
   `greedy_reference`.
3. Print the mask `run_step` builds for two sequences with 2 and 4 new
   tokens and past lengths 3 and 1. Check each row by hand.

## Next

[Section 5.2](05_02_continuous_batching.md) replaces
the one-at-a-time loop with continuous batching.

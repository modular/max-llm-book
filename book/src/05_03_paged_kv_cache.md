# 5.3 A paged KV cache

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/modular/max-llm-book/blob/main/notebooks/05_03_paged_kv_cache.ipynb)
[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/modular/max-llm-book/blob/main/notebooks/05_03_paged_kv_cache.ipynb)
[![View on GitHub](https://img.shields.io/badge/GitHub-notebook-black?logo=github)](https://github.com/modular/max-llm-book/blob/main/notebooks/05_03_paged_kv_cache.ipynb)

With continuous batching, memory limits how many requests can be in flight:
every request needs room for its keys and values, and
[Section 1.8](01_08_kv_cache.md#how-big-is-the-cache)
measured 45 KiB per token for SmolLM2-135M in float32. The question is how to
hand out that memory. A server doesn't know how long a reply will be when it
starts, so the simplest scheme reserves room for the maximum length, and most of
that room goes unused. (`ContiguousCache` avoids the reservation by copying its
arrays on every append, which costs time instead.) This section replaces the
contiguous cache with a *paged* one, the idea behind vLLM's PagedAttention and
MAX Serve's paged KV cache.

```python
import logging
import time
import warnings

import numpy as np
import plotly.graph_objects as go
from max.driver import CPU
from max.pipelines import AutoTokenizer

from minillm import viz
from minillm.device import pick_device
from minillm.config import MODEL_ID, LlamaConfig
from minillm.engine import Engine
from minillm.model import load_model
from minillm.step import Request, StepModel, compile_step
from minillm.weights import load_checkpoint

logging.getLogger("max").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="The eager interpreter failed")
DEVICE = pick_device()  # a GPU if MAX finds one, otherwise the CPU

config = LlamaConfig.from_pretrained()
tokenizer = AutoTokenizer(MODEL_ID)
step = compile_step(
    load_model(config, load_checkpoint(), DEVICE, StepModel), config, DEVICE
)
```

## Blocks and block tables

Paging works like virtual memory. The cache memory is one *pool* of
fixed-size *blocks*, each holding the keys and values of `block_size`
tokens. A sequence owns a list of block numbers, its *block table*, and
takes a new block from the pool only when its last block fills. A
sequence's blocks needn't be next to each other, so free memory never breaks
into gaps too small to use; the only waste is the unfilled part of each
sequence's last block. A finished sequence returns its blocks to the pool at
once.

```python
from __future__ import annotations

import numpy as np

from minillm.config import LlamaConfig


class BlockPool:
    """Fixed-size blocks of KV cache memory, shared by all sequences."""

    def __init__(
        self, config: LlamaConfig, num_blocks: int, block_size: int = 16
    ) -> None:
        self.block_size = block_size
        shape = (
            num_blocks,
            block_size,
            config.num_key_value_heads,
            config.head_dim,
        )
        self.keys = [
            np.zeros(shape, np.float32) for _ in range(config.num_hidden_layers)
        ]
        self.values = [
            np.zeros(shape, np.float32) for _ in range(config.num_hidden_layers)
        ]
        self.free = list(range(num_blocks))

    @property
    def num_blocks(self) -> int:
        return len(self.keys[0])

    def allocate(self) -> int:
        if not self.free:
            raise MemoryError("the KV cache pool is full")
        return self.free.pop()

    def available(self) -> int:
        """Returns how many blocks ``allocate`` can still hand out."""
        return len(self.free)

    def release(self, blocks: list[int]) -> None:
        self.free.extend(blocks)


class PagedCache:
    """One sequence's keys and values, stored in pool blocks."""

    def __init__(self, pool: BlockPool) -> None:
        self.pool = pool
        self.blocks: list[int] = []
        self.length = 0

    @property
    def num_layers(self) -> int:
        return len(self.pool.keys)

    def read(self, layer: int) -> tuple[np.ndarray, np.ndarray]:
        keys = self.pool.keys[layer][self.blocks].reshape(
            -1, *self.pool.keys[layer].shape[2:]
        )
        values = self.pool.values[layer][self.blocks].reshape(keys.shape)
        return keys[: self.length], values[: self.length]

    def append(self, layer: int, keys: np.ndarray, values: np.ndarray) -> None:
        size, start = self.pool.block_size, self.length
        if layer == 0:  # every layer stores the same positions
            while len(self.blocks) * size < start + len(keys):
                self.blocks.append(self.pool.allocate())
        for i in range(len(keys)):
            block, slot = self.blocks[(start + i) // size], (start + i) % size
            self.pool.keys[layer][block, slot] = keys[i]
            self.pool.values[layer][block, slot] = values[i]
        if layer == self.num_layers - 1:
            self.length += len(keys)

    def release(self) -> None:
        self.pool.release(self.blocks)
        self.blocks = []
```

<small>Saved to `minillm/paged.py`.</small>

`read` gathers a sequence's blocks into one array for the step model. That copy
keeps the code simple. Production engines skip it: their attention kernel reads
the blocks in place through the block table, which takes a custom kernel like
the ones [Section 3.1](03_01_custom_ops.md) shows how
to write. The engine needs three changes. It admits waiting requests in arrival
order while the pool has blocks for their prompts. When a decode step needs new
blocks and the pool is empty, it *preempts* the newest running request: it frees
that request's blocks and puts it back at the front of the queue, to recompute
its prompt and the tokens it has generated so far when it runs again. And it
returns a finished request's blocks to the pool.

```python
from minillm.engine import Engine
from minillm.step import Request


class PagedEngine(Engine):
    """An engine whose requests share one pool of KV cache blocks."""

    def __init__(self, *args, pool: BlockPool, **kwargs) -> None:
        super().__init__(*args, new_cache=lambda _: PagedCache(pool), **kwargs)
        self.pool = pool
        self.preemptions = 0

    def step(self) -> None:
        # Admit waiting requests in arrival order while their tokens so far fit.
        spare, admitted = self.pool.available(), 0
        for request in self.waiting:
            if self.blocks_needed(request) > spare:
                break
            spare -= self.blocks_needed(request)
            admitted += 1
        held = [self.waiting.pop() for _ in range(len(self.waiting) - admitted)]
        prefill = (
            self.waiting
            and len(self.running) < self.max_batch
            and (self.continuous or not self.running)
        )
        if not prefill and not self.running and held:
            raise MemoryError(
                "a waiting request needs more blocks than the pool has"
            )
        if not prefill:
            held = self.preempt_until_decode_fits() + held
        super().step()
        self.waiting.extend(held)

    def preempt_until_decode_fits(self) -> list[Request]:
        """Frees the newest requests' blocks until every running request can
        store its next token, and returns them to recompute later."""
        preempted = []
        while self.blocks_to_grow() > self.pool.available():
            if len(self.running) == 1:
                raise MemoryError(
                    "a request needs more blocks than the pool has"
                )
            request, cache = self.running.pop()
            cache.release()
            preempted.insert(0, request)
            self.preemptions += 1
        return preempted

    def blocks_to_grow(self) -> int:
        size = self.pool.block_size
        return sum(c.length + 1 > len(c.blocks) * size for _, c in self.running)

    def blocks_needed(self, request: Request) -> int:
        # A preempted request recomputes the tokens it already generated.
        tokens = len(request.prompt) + len(request.output)
        return -(-tokens // self.pool.block_size)

    def prefill_tokens(self, request: Request, cache: PagedCache) -> list[int]:
        return request.prompt + request.output

    def release_finished(self, batch: list) -> None:
        for request, cache in batch:
            if request.done:
                cache.release()
```

<small>Saved to `minillm/paged.py`.</small>

## Check it

Paging only changes where the keys and values are stored. Serve the workload
from [Section 5.2](05_02_continuous_batching.md) with
a paged cache and compare the tokens:

```python
TEXTS = [
    "Tell me about Paris.",
    "Write a haiku about the sea.",
    "What is 2 + 2?",
    "List three colors.",
    "Explain gravity in one sentence.",
    "Name a fruit.",
    "Describe a cat.",
    "Say hello in French.",
]
LENGTHS = [32, 24, 8, 16, 32, 8, 24, 16]


def workload() -> list[Request]:
    now = time.perf_counter()
    return [
        Request(tokenizer([t])["input_ids"][0], n, arrived=now)
        for t, n in zip(TEXTS, LENGTHS)
    ]


contiguous_requests = workload()
engine = Engine(step, config, DEVICE, max_batch=4)
for r in contiguous_requests:
    engine.add(r)
engine.run()

pool = BlockPool(config, num_blocks=32, block_size=16)
paged_requests = workload()
paged = PagedEngine(step, config, DEVICE, max_batch=4, pool=pool)
for r in paged_requests:
    paged.add(r)
paged.run()

print(
    "same tokens:",
    [r.output for r in paged_requests]
    == [r.output for r in contiguous_requests],
)
print("blocks free after the run:", len(pool.free), "of", pool.num_blocks)
assert [r.output for r in paged_requests] == [
    r.output for r in contiguous_requests
]
assert len(pool.free) == pool.num_blocks
```

```text
same tokens: True
blocks free after the run: 32 of 32
```

The tokens match, and every block is back in the pool after the run. To
watch the blocks move, run the workload again on a smaller pool of 24
blocks of 8 tokens, and record after every engine step which request owns
each block. Slide through the steps:

```python
class RecordingEngine(PagedEngine):
    def step(self) -> None:
        super().step()
        owner = np.full(self.pool.num_blocks, -1)
        for request, cache in self.running:
            owner[cache.blocks] = requests.index(request)
        self.snapshots.append(owner)


requests = workload()
recorder = RecordingEngine(
    step, config, DEVICE, max_batch=4, pool=BlockPool(config, 24, 8)
)
recorder.snapshots = []
for r in requests:
    recorder.add(r)
recorder.run()
print("preemptions:", recorder.preemptions)

colors = [viz.GRID] + viz.SERIES  # free, then one color per request
scale = [
    [edge, c] for i, c in enumerate(colors) for edge in (i / 9, (i + 1) / 9)
]
frames = []
for owner in recorder.snapshots:
    grid = owner.reshape(3, 8)
    text = [["" if o < 0 else f"r{o}" for o in row] for row in grid]
    hover = [
        [
            f"block {r * 8 + c}: " + ("free" if o < 0 else f"request {o}")
            for c, o in enumerate(row)
        ]
        for r, row in enumerate(grid)
    ]
    frames.append(
        [
            go.Heatmap(
                z=grid,
                zmin=-1.5,
                zmax=7.5,
                colorscale=scale,
                showscale=False,
                xgap=3,
                ygap=3,
                text=text,
                texttemplate="%{text}",
                customdata=hover,
                hovertemplate="%{customdata}<extra></extra>",
            )
        ]
    )
fig = go.Figure(
    layout=dict(
        title="Who owns each of the pool's 24 blocks",
        height=380,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, autorange="reversed"),
    )
)
labels = [f"{i + 1}" for i in range(len(frames))]
viz.show(
    viz.with_slider(
        fig, frames, labels, "after engine step ", active=len(frames) // 3
    )
)
```

```text
preemptions: 0
```

<div class="plotly-figure" data-src="generated/05_03_paged_kv_cache_10_1.json"></div>

Blocks are handed out from wherever they are free, so a request's blocks
end up scattered across the pool, and a finished request's blocks go
straight to the next request. Only the block table records their order.
This pool was big enough that no request was preempted; exercise 1 shrinks
it until some are.

## Why it matters: memory

Compare the memory each scheme ties up for a batch of four of these
requests. A contiguous cache can't grow in place, so when a request starts
it must reserve room for the longest the request could get. That is
SmolLM2's full 8192-token context, or at best the request's `max_tokens`
limit, which API clients often set well above the length of the replies they
get; the table below assumes 512. Paging allocates only what each sequence
holds, rounded up to whole blocks of \\(B\\) tokens:

\\[\\text{blocks for } n \\text{ tokens} = \\left\\lceil \\frac{n}{B} \\right\\rceil\\]

```python
per_token = (
    2
    * config.num_hidden_layers
    * config.num_key_value_heads
    * config.head_dim
    * 4
)
batch = 4
actual = [len(r.prompt) + len(r.output) for r in paged_requests]
reserve_max_context = batch * config.max_position_embeddings
reserve_max_tokens = batch * (max(len(r.prompt) for r in paged_requests) + 512)
paged_tokens = sum(sorted(-(-n // 16) * 16 for n in actual)[-batch:])

print(f"{'scheme':38} {'tokens reserved':>15} {'memory':>9}")
for name, tokens in [
    ("reserve full context per request", reserve_max_context),
    ("reserve max_tokens=512 per request", reserve_max_tokens),
    ("paged, 16-token blocks (4 largest)", paged_tokens),
]:
    print(f"{name:38} {tokens:>15,} {tokens * per_token / 2**20:>7.1f} MiB")

schemes = {
    "reserve full context": reserve_max_context,
    "reserve max_tokens=512": reserve_max_tokens,
    "paged, 16-token blocks": paged_tokens,
}
fig = go.Figure(
    go.Bar(
        y=list(schemes),
        x=[t * per_token / 2**20 for t in schemes.values()],
        orientation="h",
        marker_color=[viz.ORANGE, viz.YELLOW, viz.BLUE],
        text=[f"{t * per_token / 2**20:,.1f} MiB" for t in schemes.values()],
        textposition="outside",
        cliponaxis=False,
        hovertemplate="%{y}: %{x:,.1f} MiB<extra></extra>",
    )
)
fig.update_layout(
    title="KV cache memory for a batch of 4 of these requests",
    height=300,
    margin=dict(r=96),
    xaxis=dict(title="MiB (log scale)", type="log", dtick=1),
    yaxis=dict(autorange="reversed", showgrid=False),
)
viz.show(fig)
```

```text
scheme                                 tokens reserved    memory
reserve full context per request                32,768  1440.0 MiB
reserve max_tokens=512 per request               2,080    91.4 MiB
paged, 16-token blocks (4 largest)                 160     7.0 MiB
```

<div class="plotly-figure" data-src="generated/05_03_paged_kv_cache_12_1.json"></div>

Saved memory becomes batch capacity. In the memory that one full-context
reservation needs (8192 tokens), a paged cache fits about two hundred of
these requests at roughly 40 tokens each, which is why most serving engines
page their KV cache. The block size is a trade-off: bigger blocks waste more
space in each sequence's last block, and smaller ones mean longer block
tables and more bookkeeping per step.

MAX Serve's paged cache works the same way, and its attention kernels read the
pages through the block table directly. Its scheduler also preempts a running
request when the pool runs out of pages, and logs a warning that suggests giving
the cache more memory. `max serve` exposes the settings. `--kv-cache-page-size`
sets the block size, 128 tokens by default. `--device-memory-utilization` sets
the fraction of device memory the process may use, 0.95 by default, and the KV
cache pool gets what is left after the weights and other allocations. The pool
size and `--max-batch-size` together limit how many requests run at once. The
handbook's
[PagedAttention](https://handbook.modular.com/inference-optimization/pagedattention/)
page describes the algorithm, and
[GPU memory for LLMs](https://handbook.modular.com/getting-started/calculating-gpu-memory-for-llms/)
shows how to size the pool.

## Exercises

1. Shrink the pool to 8 blocks and rerun the paged engine. How does the
   engine behave when memory runs out, and what happens to time to first
   token?
2. Try block sizes 4, 16, and 64. How much memory goes to partly filled last
   blocks in each case?
3. This engine preempts the newest request. Change it to preempt the
   request that holds the most blocks instead. Which choice finishes the
   workload in fewer steps, and which treats long requests more fairly?

## Next

[Section 5.4](05_04_prefix_caching.md) lets
requests that share a prompt prefix share its blocks too.

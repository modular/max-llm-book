# 5.4 Prefix caching

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/modular/max-llm-book/blob/main/notebooks/05_04_prefix_caching.ipynb)
[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/modular/max-llm-book/blob/main/notebooks/05_04_prefix_caching.ipynb)
[![View on GitHub](https://img.shields.io/badge/GitHub-notebook-black?logo=github)](https://github.com/modular/max-llm-book/blob/main/notebooks/05_04_prefix_caching.ipynb)

Real traffic repeats itself. Requests to a chat product start with the same
system prompt, and every turn of a conversation resends the turns before it.
Those shared tokens get the same keys and values every time, because a token's
keys and values depend only on that token and the tokens before it. *Prefix
caching* computes them once and shares their cache blocks between requests. With
the paged cache from
[Section 5.3](05_03_paged_kv_cache.md), it takes
little code.

```python
import logging
import time
import warnings
from collections import OrderedDict

import numpy as np
import plotly.graph_objects as go
from max.driver import CPU
from max.pipelines import AutoTokenizer

from minillm import viz
from minillm.device import pick_device
from minillm.config import MODEL_ID, LlamaConfig
from minillm.model import load_model
from minillm.paged import BlockPool, PagedCache, PagedEngine
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

## Naming blocks by their content

To share a block, the engine must recognize it. A full block is identified
by the tokens it holds *and every token before it*, since those determine
its keys and values. A chained hash captures both: each block's hash
combines the previous block's hash with the block's own tokens. Two requests
share a block exactly when their prompts agree up to the end of that block.

The pool keeps a table from hash to block, counts how many sequences use
each block, and frees a block only when no sequence uses it. A block with no
users whose contents are still valid stays in the table, in
least-recently-used order, so a later request can still reuse it. Such
blocks are evicted only when the pool runs out of free blocks.

Only *full* blocks are shared, and new tokens always go into a sequence's
own next block, so a shared block is never written to.

```python
from __future__ import annotations

from collections import OrderedDict

from minillm.config import LlamaConfig
from minillm.paged import BlockPool, PagedCache, PagedEngine
from minillm.step import Request


class PrefixCachingPool(BlockPool):
    """A block pool that remembers full prompt blocks for reuse."""

    def __init__(
        self, config: LlamaConfig, num_blocks: int, block_size: int = 16
    ) -> None:
        super().__init__(config, num_blocks, block_size)
        self.refs = [0] * num_blocks
        self.by_hash: dict[int, int] = {}
        self.hash_of: dict[int, int] = {}
        self.unused: OrderedDict[int, None] = OrderedDict()  # cached, no users

    def allocate(self) -> int:
        if self.free:
            block = self.free.pop()
        elif self.unused:
            block, _ = self.unused.popitem(last=False)  # evict the oldest
            del self.by_hash[self.hash_of.pop(block)]
        else:
            raise MemoryError("the KV cache pool is full")
        self.refs[block] = 1
        return block

    def available(self) -> int:
        # Cached blocks that no request uses can be evicted and reused.
        return len(self.free) + len(self.unused)

    def release(self, blocks: list[int]) -> None:
        for block in blocks:
            self.refs[block] -= 1
            if self.refs[block] == 0:
                if block in self.hash_of:
                    self.unused[block] = None
                else:
                    self.free.append(block)

    def block_hashes(self, tokens: list[int]) -> list[int]:
        hashes, previous = [], 0
        for start in range(
            0, len(tokens) - self.block_size + 1, self.block_size
        ):
            previous = hash(
                (previous, tuple(tokens[start : start + self.block_size]))
            )
            hashes.append(previous)
        return hashes

    def lookup(self, prompt: list[int]) -> list[int]:
        """Returns the cached blocks that start ``prompt``, and claims them.
        The last prompt token is always left to compute, since its logits
        are needed."""
        blocks = []
        for h in self.block_hashes(prompt[:-1]):
            block = self.by_hash.get(h)
            if block is None:
                break
            self.refs[block] += 1
            self.unused.pop(block, None)
            blocks.append(block)
        return blocks

    def register(self, prompt: list[int], blocks: list[int]) -> None:
        """Records a sequence's full prompt blocks for later requests."""
        # The last block may be partial; only full blocks have hashes.
        for h, block in zip(self.block_hashes(prompt), blocks, strict=False):
            if h not in self.by_hash:
                self.by_hash[h] = block
                self.hash_of[block] = h


class PrefixCachingEngine(PagedEngine):
    """A paged engine that skips computing prompt prefixes it has cached."""

    def __init__(self, *args, pool: PrefixCachingPool, **kwargs) -> None:
        super().__init__(*args, pool=pool, **kwargs)
        self.computed_prompt_tokens = 0

    def prefill_tokens(self, request: Request, cache: PagedCache) -> list[int]:
        tokens = (
            request.prompt + request.output
        )  # output is empty unless preempted
        cache.blocks = self.pool.lookup(tokens)
        cache.length = len(cache.blocks) * self.pool.block_size
        new = tokens[cache.length :]
        self.computed_prompt_tokens += len(new)
        return new

    def prefilled(self, request: Request, cache: PagedCache) -> None:
        self.pool.register(request.prompt, cache.blocks)
```

<small>Saved to `minillm/prefix.py`.</small>

`prefill_tokens` and `prefilled` are the two hooks the engine from
[Section 5.2](05_02_continuous_batching.md) calls
around each prefill. The base engine returns the whole prompt from the first and
does nothing in the second. The step model already handles the rest: a prefill
with a non-empty past is a step with several new tokens.

## A workload with a shared prefix

Eight users ask different questions of the same assistant. Its system
prompt is about 150 tokens long; real ones are often much longer:

```python
SYSTEM = (
    "You are a helpful assistant for a travel company. Answer briefly and politely. "
    "Recommend destinations in Europe, mention the best season to visit, give one "
    "practical tip about transport, and never invent prices. "
) * 4
QUESTIONS = [
    "Where should I go in spring?",
    "Is Rome good in winter?",
    "How do I get around Paris?",
    "What about Lisbon?",
    "Best city for museums?",
    "Where can I hike?",
    "Is Berlin expensive?",
    "Suggest a beach town.",
]


def workload() -> list[Request]:
    now = time.perf_counter()
    return [
        Request(tokenizer([SYSTEM + q])["input_ids"][0], 8, arrived=now)
        for q in QUESTIONS
    ]


print("prompt tokens per request:", len(workload()[0].prompt))


def serve(engine) -> list[Request]:
    requests = workload()
    for r in requests:
        engine.add(r)
    start = time.perf_counter()
    engine.run()
    ttft = [(r.first_token - r.arrived) * 1000 for r in requests]
    print(
        f"{type(engine).__name__:20} {time.perf_counter() - start:5.1f} s   "
        f"TTFT median {np.median(ttft):5.0f} ms, worst {max(ttft):5.0f} ms"
    )
    return requests


class CountingPrefixEngine(PrefixCachingEngine):
    """Records how many prompt tokens each request computes."""

    def prefill_tokens(self, request: Request, cache: object) -> list[int]:
        tokens = super().prefill_tokens(request, cache)
        self.computed_per_request[id(request)] = len(tokens)
        return tokens


plain = serve(
    PagedEngine(step, config, DEVICE, max_batch=4, pool=BlockPool(config, 128))
)
prefix_pool = PrefixCachingPool(config, 128)
cached_engine = CountingPrefixEngine(
    step, config, DEVICE, max_batch=4, pool=prefix_pool
)
cached_engine.computed_per_request = {}
cached = serve(cached_engine)
```

```text
prompt tokens per request: 163
PagedEngine            6.4 s   TTFT median  2143 ms, worst  3812 ms
CountingPrefixEngine   5.6 s   TTFT median  1774 ms, worst  3106 ms
```

```python
total_prompt = sum(len(r.prompt) for r in cached)
print(
    f"prompt tokens computed: {cached_engine.computed_prompt_tokens} of {total_prompt} "
    f"({cached_engine.computed_prompt_tokens / total_prompt:.0%})"
)
print("same tokens:", [r.output for r in cached] == [r.output for r in plain])
assert [r.output for r in cached] == [r.output for r in plain]
assert cached_engine.computed_prompt_tokens < total_prompt

computed = [cached_engine.computed_per_request[id(r)] for r in cached]
reused = [len(r.prompt) - c for r, c in zip(cached, computed)]
names = [f"request {i}: {q}" for i, q in enumerate(QUESTIONS)]
fig = go.Figure(
    [
        go.Bar(
            y=names,
            x=reused,
            name="reused from the cache",
            orientation="h",
            marker_color=viz.BLUE,
            hovertemplate="%{x} tokens reused<extra>%{y}</extra>",
        ),
        go.Bar(
            y=names,
            x=computed,
            name="computed",
            orientation="h",
            marker_color=viz.ORANGE,
            hovertemplate="%{x} tokens computed<extra>%{y}</extra>",
        ),
    ]
)
fig.update_layout(
    title="Prompt tokens per request with prefix caching",
    barmode="stack",
    height=420,
    xaxis_title="prompt tokens",
    yaxis=dict(autorange="reversed", showgrid=False),
)
viz.show(fig)
```

```text
prompt tokens computed: 715 of 1291 (55%)
same tokens: True
```

<div class="plotly-figure" data-src="generated/05_04_prefix_caching_7_1.json"></div>

The first batch computes the system prompt once per request, because nothing is
cached yet. From the second batch on, each request finds the system prompt's
full blocks in the table and computes only the rest: the system prompt's last,
partly filled block plus its own question. That brings the prompt tokens
computed down to 715 of 1,291, or 55%. The outputs are identical, because the
reused keys and values are exactly the ones the request would have computed.

The wall-clock gain is modest here: the whole workload took about 5.6 s with
prefix caching against 6.4 s without, on this page's machine (an Apple-silicon
Mac with a Metal GPU), because prefilling about 160 prompt tokens on a 135M
model is only part of the workload's time. Treat these numbers as an example:
yours depend on your hardware and vary from run to run. The saving grows with
the shared prefix. A multi-turn chat resends the whole conversation so far on
every turn, often thousands of tokens, and an agent resends long tool
descriptions on every call, so for them prefix caching removes most of the
prefill.

MAX Serve enables prefix caching by default; `--no-enable-prefix-caching` turns
it off. The same idea extends across servers: a router can send each request to
the replica that already holds its prefix. The handbook covers
[prefix caching](https://handbook.modular.com/inference-optimization/prefix-caching/)
and
[inference routing](https://handbook.modular.com/inference-optimization/inference-routing/).

## Exercises

1. Send the workload twice through the same engine. What fraction of prompt
   tokens does the second pass compute?
2. Shrink the pool so cached blocks must be evicted. Which requests lose
   their prefix, and why does least-recently-used eviction suit chat traffic?
3. Why does `lookup` hash `prompt[:-1]` instead of the whole prompt? What
   would break otherwise?

## Next

[Section 5.5](05_05_speculative_decoding.md) speeds up
decoding instead: speculative decoding can produce several tokens per step of
the large model.

# 5.2 Continuous batching

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/modular/max-llm-book/blob/main/notebooks/05_02_continuous_batching.ipynb)
[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/modular/max-llm-book/blob/main/notebooks/05_02_continuous_batching.ipynb)
[![View on GitHub](https://img.shields.io/badge/GitHub-notebook-black?logo=github)](https://github.com/modular/max-llm-book/blob/main/notebooks/05_02_continuous_batching.ipynb)

The engine from
[Section 5.1](05_01_request_lifecycle.md) ran one
sequence per step, while
[Section 4.3](04_03_kernel_benchmarks.md#benchmark-a-layer)
showed that a decode step for many sequences costs little more than a step for
one. Batching uses that spare capacity, and the way you batch decides who waits.
This section adds a scheduler to the engine and compares three policies on the
same workload.

```python
import logging
import time
import warnings
from collections import deque

import numpy as np
import plotly.graph_objects as go
from max.driver import CPU
from max.pipelines import AutoTokenizer

from minillm import viz
from minillm.device import pick_device
from minillm.config import MODEL_ID, LlamaConfig
from minillm.model import load_model
from minillm.step import (
    ContiguousCache,
    Request,
    StepModel,
    compile_step,
    run_step,
)
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

## The engine

The engine holds two lists: requests *waiting* to start and requests
*running*. Each call to `step` does one model step:

- If there is room in the batch and requests are waiting, it admits them
  and runs their *prefill*: all their prompt tokens, in one batched step.
- Otherwise it runs a *decode* step: one new token for every running
  request, in one batched step.

Finished requests leave the batch after every step. That is *continuous*
batching: the batch is re-formed at every step, so a waiting request starts
as soon as a slot frees up. Setting `continuous=False` gives *static*
batching instead: a batch is admitted together and must finish completely
before the next one starts.

Running prefill and decode in separate steps is a simple policy that gives
new requests priority, at the cost of pausing the running requests' decode
for a step. Production engines can also split a long prompt into chunks
(*chunked prefill*) and run prefill chunks and decodes in the same step, so
one big prefill doesn't stall everyone. MAX Serve chunks long prefills by
default (`--enable-chunked-prefill`) and batches them with decodes when you
pass `--enable-in-flight-batching`.

```python
from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable

from max.driver import Device

from minillm.config import LlamaConfig
from minillm.step import ContiguousCache, Request, Step, run_step


class Engine:
    """Serves requests with a batching scheduler and greedy decoding."""

    def __init__(
        self,
        step: Step,
        config: LlamaConfig,
        device: Device,
        max_batch: int = 4,
        continuous: bool = True,
        new_cache: Callable[[LlamaConfig], object] = ContiguousCache,
    ) -> None:
        self.step_fn, self.config, self.device = step, config, device
        self.max_batch, self.continuous, self.new_cache = (
            max_batch,
            continuous,
            new_cache,
        )
        self.waiting: deque[Request] = deque()
        self.running: list[tuple[Request, object]] = []
        self.steps = 0

    def add(self, request: Request) -> None:
        self.waiting.append(request)

    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    def step(self) -> None:
        room = self.max_batch - len(self.running)
        can_admit = (
            self.waiting and room > 0 and (self.continuous or not self.running)
        )
        if can_admit:
            admitted = [
                self.waiting.popleft()
                for _ in range(min(room, len(self.waiting)))
            ]
            batch = [(r, self.new_cache(self.config)) for r in admitted]
            inputs = [self.prefill_tokens(r, c) for r, c in batch]
        else:
            batch = self.running
            inputs = [[r.output[-1]] for r, _ in batch]
        logits = run_step(
            self.step_fn, [c for _, c in batch], inputs, self.device
        )
        self.steps += 1
        now = time.perf_counter()
        for (request, _), token in zip(batch, logits.argmax(-1), strict=True):
            request.output.append(int(token))
            if request.first_token is None:
                request.first_token = now
            if request.done:
                request.finished = now
        if can_admit:
            for request, cache in batch:
                self.prefilled(request, cache)
            self.running += batch
        self.running = [(r, c) for r, c in self.running if not r.done]
        self.release_finished(batch)

    def prefill_tokens(self, request: Request, cache: object) -> list[int]:
        """Returns the prompt tokens to compute; Section 5.4 skips cached ones."""
        return request.prompt

    def prefilled(self, request: Request, cache: object) -> None:
        """Runs after a request's prefill; Section 5.4 uses it."""

    def release_finished(self, batch: list) -> None:
        """Frees the caches of finished requests; Section 5.3 needs this."""

    def run(self) -> None:
        while self.has_work():
            self.step()
```

<small>Saved to `minillm/engine.py`.</small>

## A workload

Eight requests arrive at once, each asking for between 8 and 32 new tokens.
The workload is the same as in
[Section 5.1](05_01_request_lifecycle.md):

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


def run_policy(
    max_batch: int, continuous: bool
) -> tuple[list[Request], Engine]:
    engine = Engine(
        step, config, DEVICE, max_batch=max_batch, continuous=continuous
    )
    now = time.perf_counter()
    requests = [
        Request(tokenizer([t])["input_ids"][0], n, arrived=now)
        for t, n in zip(TEXTS, LENGTHS)
    ]
    for r in requests:
        engine.add(r)
    engine.run()
    return requests, engine


def summarize(name: str, requests: list[Request], engine: Engine) -> dict:
    ttft = [(r.first_token - r.arrived) * 1000 for r in requests]
    total = max(r.finished for r in requests) - min(r.arrived for r in requests)
    tokens = sum(len(r.output) for r in requests)
    print(
        f"{name:22} {engine.steps:4} steps  {tokens / total:5.1f} tokens/s  "
        f"TTFT median {np.median(ttft):6.0f} ms  worst {max(ttft):6.0f} ms"
    )
    return {
        "ttft": ttft,
        "outputs": [r.output for r in requests],
        "requests": requests,
    }


results = {}
for name, max_batch, continuous in [
    ("one at a time", 1, True),
    ("static batches of 4", 4, False),
    ("continuous, batch 4", 4, True),
]:
    results[name] = summarize(name, *run_policy(max_batch, continuous))
```

```text
one at a time           160 steps    3.6 tokens/s  TTFT median  20020 ms  worst  39529 ms
static batches of 4      64 steps    8.5 tokens/s  TTFT median   4942 ms  worst   9643 ms
continuous, batch 4      50 steps   11.3 tokens/s  TTFT median   1334 ms  worst   7702 ms
```

Batching changes only the order of the work, so greedy decoding must
produce the same tokens for every request under every policy:

```python
reference = results["one at a time"]["outputs"]
print(all(r["outputs"] == reference for r in results.values()))
assert all(r["outputs"] == reference for r in results.values())
```

```text
True
```

```python
steps = []
for name, r in results.items():
    start = min(q.arrived for q in r["requests"])
    rows = [f"request {i} ({n} tokens)" for i, n in enumerate(LENGTHS)]
    wait = [(q.first_token - q.arrived) * 1000 for q in r["requests"]]
    gen = [(q.finished - q.first_token) * 1000 for q in r["requests"]]
    first = [(q.first_token - start) * 1000 for q in r["requests"]]
    steps.append(
        [
            go.Bar(
                y=rows,
                x=wait,
                base=[(q.arrived - start) * 1000 for q in r["requests"]],
                orientation="h",
                name="waiting, then prefill",
                marker_color=viz.MUTED,
                legendgroup="wait",
                hovertemplate="%{y}: first token after %{x:.0f} ms<extra></extra>",
            ),
            go.Bar(
                y=rows,
                x=gen,
                base=first,
                orientation="h",
                name="generating",
                marker_color=viz.BLUE,
                legendgroup="gen",
                hovertemplate="%{y}: generating for %{x:.0f} ms<extra></extra>",
            ),
        ]
    )
fig = go.Figure(
    layout=dict(
        title="Each request's life under each policy",
        height=460,
        barmode="overlay",
        xaxis_title="milliseconds since the requests arrived",
        yaxis=dict(autorange="reversed", showgrid=False),
    )
)
viz.show(viz.with_slider(fig, steps, list(results), "policy: ", active=2))
```

<div class="plotly-figure" data-src="generated/05_02_continuous_batching_9_0.json"></div>

Slide between the policies. One at a time, each request waits for every
request before it. Static batches run in two rounds, and the second round
can't start until the slowest request of the first finishes. Continuous
batching starts a waiting request as soon as any slot frees up.

The same data as a distribution of time to first token:

```python
fig = go.Figure(
    [
        go.Scatter(
            x=sorted(r["ttft"]),
            y=np.arange(1, 9) / 8,
            name=name,
            mode="lines+markers",
            line=dict(color=color, shape="hv"),
            hovertemplate=name
            + ": %{y:.0%} of requests within %{x:.0f} ms<extra></extra>",
        )
        for (name, r), color in zip(results.items(), viz.SERIES)
    ]
)
fig.update_layout(
    title="Time to first token across the 8 requests",
    height=400,
    xaxis_title="time to first token (ms)",
    yaxis=dict(title="fraction of requests", tickformat=".0%", range=[0, 1.05]),
)
viz.show(fig)
```

<div class="plotly-figure" data-src="generated/05_02_continuous_batching_11_0.json"></div>

Read the second chart as "what fraction of users had their first token
within this time". Both batching policies cut the number of model steps,
from 160 to 64 for static batches and 50 for continuous batching, so they
finish the workload sooner and raise throughput. They differ in who waits:

- Static batching admits four requests and runs until the *longest* of
  them finishes. The second group waits for that, even after the short
  requests in the first group are done and their slots sit empty.
- Continuous batching refills a slot as soon as a request finishes, so
  waiting requests start sooner, and the batch stays full while any request
  is waiting.

Production engines, MAX Serve among them, use continuous batching with much
larger batches, bounded by the memory the KV cache needs. That bound is the
subject of [Section 5.3](05_03_paged_kv_cache.md). The
handbook's
[static, dynamic, and continuous batching](https://handbook.modular.com/inference-optimization/static-dynamic-continuous-batching/)
page compares the policies in production terms.

## Exercises

1. Run continuous batching with `max_batch` 2, 4, and 8. How do throughput
   and worst-case TTFT change?
2. Make requests arrive over time: add each one after a few engine steps
   instead of all at the start. Which policy handles bursts better?
3. Implement a mixed step: admit new requests' prefills in the *same* step
   as running requests' decodes. What does it cost the running requests?

## Next

[Section 5.3](05_03_paged_kv_cache.md) replaces the
contiguous cache with a paged one, so the batch can grow without wasting memory.

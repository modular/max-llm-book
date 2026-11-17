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

<details class="code-fold">
<summary>Show the setup <span class="code-fold-lines">31 lines</span></summary>

```python
import logging
import warnings

import numpy as np
import plotly.graph_objects as go
from max.pipelines import AutoTokenizer

from minillm import viz
from minillm.config import MODEL_ID, LlamaConfig
from minillm.device import pick_device
from minillm.model import load_model
from minillm.step import (
    WORKLOAD,
    ContiguousCache,
    Request,
    StepModel,
    compile_step,
    make_workload,
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

</details>

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

<details class="code-fold">
<summary>Show <code>Engine</code> <span class="code-fold-lines">86 lines</span></summary>

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
        self.step_fn = step
        self.config = config
        self.device = device
        self.max_batch = max_batch
        self.continuous = continuous
        self.new_cache = new_cache
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

</details>

<small>Saved to `minillm/engine.py`.</small>

## A workload

Serve the eight requests from
[Section 5.1](05_01_request_lifecycle.md) under three
policies. A recording subclass of `Engine` notes which request holds each batch
slot at every step, for the figures below:

<details class="code-fold">
<summary>Show <code>RecordingEngine</code>, <code>run_policy</code>, <code>summarize</code> <span class="code-fold-lines">51 lines</span></summary>

```python
class RecordingEngine(Engine):
    """Records, per step, the request in each batch slot and what it did."""

    def step(self) -> None:
        before = {id(r): len(r.output) for r in self.requests}
        super().step()
        ran = [r for r in self.requests if len(r.output) > before[id(r)]]
        for r in ran:
            if r not in self.slots:
                self.slots[self.slots.index(None)] = r
        row = []
        for r in self.slots:
            if r is None:
                row.append(None)
            elif r not in ran:
                row.append((self.requests.index(r), "waits"))
            elif before[id(r)] == 0:
                row.append((self.requests.index(r), "prefill"))
            else:
                row.append((self.requests.index(r), "decode"))
        self.history.append(row)
        self.slots = [None if r is None or r.done else r for r in self.slots]


def run_policy(max_batch: int, continuous: bool) -> tuple[list, Engine]:
    engine = RecordingEngine(
        step, config, DEVICE, max_batch=max_batch, continuous=continuous
    )
    engine.requests = make_workload(tokenizer)
    engine.slots = [None] * max_batch
    engine.history = []
    for r in engine.requests:
        engine.add(r)
    engine.run()
    return engine.requests, engine


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
        "history": engine.history,
    }
```

</details>

```python
results = {}
for name, max_batch, continuous in [
    ("one at a time", 1, True),
    ("static batches of 4", 4, False),
    ("continuous, batch 4", 4, True),
]:
    results[name] = summarize(name, *run_policy(max_batch, continuous))
```

```text
one at a time           160 steps    2.9 tokens/s  TTFT median  27801 ms  worst  50743 ms
static batches of 4      64 steps    3.2 tokens/s  TTFT median   6450 ms  worst  12693 ms
continuous, batch 4      50 steps    9.3 tokens/s  TTFT median   2151 ms  worst   8903 ms
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

<details class="code-fold">
<summary>Show the code <span class="code-fold-lines">41 lines</span></summary>

```python
steps = []
for name, r in results.items():
    start = min(q.arrived for q in r["requests"])
    rows = [f"request {i} ({n} tokens)" for i, (_, n) in enumerate(WORKLOAD)]
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

</details>

<div class="plotly-figure" data-src="generated/05_02_continuous_batching_10_0.json"></div>

Slide between the policies. One at a time, each request waits for every
request before it. Static batches run in two rounds, and the second round
can't start until the slowest request of the first finishes. Continuous
batching starts a waiting request as soon as any slot frees up.

The same runs from the engine's side: each column is one engine step, each row
one of the four batch slots, and the color says which request holds the slot.
"P" marks a request's prefill step, and a dot marks a running request that sat
out the step because another request was prefilling:

<details class="code-fold">
<summary>Show the code <span class="code-fold-lines">54 lines</span></summary>

```python
policies = ["static batches of 4", "continuous, batch 4"]
colors = [viz.GRID] + viz.SERIES  # empty, then one color per request
scale = [
    [edge, c] for i, c in enumerate(colors) for edge in (i / 9, (i + 1) / 9)
]
steps = []
for name in policies:
    history = np.array(
        [
            [-1 if s is None else s[0] for s in row]
            for row in results[name]["history"]
        ]
    ).T
    marks = {"prefill": "P", "waits": "·", "decode": ""}
    text = [
        ["" if s is None else marks[s[1]] for s in row]
        for row in results[name]["history"]
    ]
    hover = [
        [
            f"step {t + 1}, slot {k}: "
            + ("empty" if s is None else f"request {s[0]}, {s[1]}")
            for t, s in enumerate(column)
        ]
        for k, column in enumerate(zip(*results[name]["history"]))
    ]
    steps.append(
        [
            go.Heatmap(
                z=history,
                x=np.arange(1, history.shape[1] + 1),
                y=[f"slot {k}" for k in range(history.shape[0])],
                zmin=-1.5,
                zmax=7.5,
                colorscale=scale,
                showscale=False,
                xgap=1,
                ygap=3,
                text=np.array(text).T,
                texttemplate="%{text}",
                customdata=hover,
                hovertemplate="%{customdata}<extra></extra>",
            )
        ]
    )
fig = go.Figure(
    layout=dict(
        title="Which request holds each batch slot at every engine step",
        height=320,
        xaxis=dict(title="engine step", showgrid=False),
        yaxis=dict(autorange="reversed", showgrid=False),
    )
)
viz.show(viz.with_slider(fig, steps, policies, "policy: ", active=1))
```

</details>

<div class="plotly-figure" data-src="generated/05_02_continuous_batching_12_0.json"></div>

Both batching policies cut the number of model steps, from 160 to 64 for
static batches and 50 for continuous batching, so they finish the workload
sooner and raise throughput. They differ in who waits:

- Static batching admits four requests and runs until the *longest* of
  them finishes. The second group waits for that, even after the short
  requests in the first group are done and their slots sit empty: the grey
  holes in the slot chart.
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

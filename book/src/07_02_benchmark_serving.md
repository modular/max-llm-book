# 7.2 Benchmark a server

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/modular/max-llm-book/blob/main/notebooks/07_02_benchmark_serving.ipynb)
[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/modular/max-llm-book/blob/main/notebooks/07_02_benchmark_serving.ipynb)
[![View on GitHub](https://img.shields.io/badge/GitHub-notebook-black?logo=github)](https://github.com/modular/max-llm-book/blob/main/notebooks/07_02_benchmark_serving.ipynb)

A server that answers one request correctly can still fall over under real
traffic. `max benchmark` measures a running server the way users experience it:
it sends many requests, some at the same time, and records how long each one
waits for its first token and for each token after that. This section benchmarks
MAX's SmolLM2 and the architecture you served in
[Section 7.1](07_01_serve_your_model.md), and explains
the numbers.

<details class="code-fold">
<summary>Show the setup <span class="code-fold-lines">15 lines</span></summary>

```python
import json
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from minillm import viz
from minillm.config import MODEL_ID
from minillm.server import start_max_serve

WORKDIR = Path(tempfile.mkdtemp())
```

</details>

## What to measure

These numbers describe a serving run. The handbook's
[LLM inference metrics](https://handbook.modular.com/llm-inference-basics/llm-inference-metrics/)
page defines them in detail:

- Time to first token (TTFT): the time from sending a request to receiving
  its first token. It includes the wait in the queue and the prefill. Users
  notice it as the delay before the answer starts.
- Time per output token (TPOT): the average gap between later tokens, one
  decode step. Users notice it as the reading speed.
- Requests per second and output tokens per second: the throughput of the
  whole server, which decides how many users one GPU can serve.

*Concurrency*, the number of requests in flight at once, connects them. More
concurrency lets the server batch more requests per step
([Chapter 5](ch05.md)), which raises throughput, but
each request can then wait longer.

## Run a benchmark

From a terminal, a benchmark is one command against a running server. This
one sends 32 requests of 128 random input tokens each, asks for 32 output
tokens per request, and keeps at most 8 in flight:

```bash
max benchmark --model HuggingFaceTB/SmolLM2-135M-Instruct --dataset-name random \
  --random-input-len 128 --random-output-len 32 \
  --num-prompts 32 --max-concurrency 8 --result-filename result.json
```

The `random` dataset gives every run the same shape, which suits
comparisons. The default dataset, `sharegpt`, replays real chat
conversations with varied lengths. The helpers below start a server, run a
benchmark from Python with the same flags, and read the result file, a JSON
object with the summary metrics (`request_throughput`, `mean_ttft_ms`,
`p95_ttft_ms`, and so on) and one record per request (`request_records`):

<details class="code-fold">
<summary>Show <code>server</code>, <code>benchmark</code>, <code>summary</code>, <code>show</code> <span class="code-fold-lines">67 lines</span></summary>

```python
@contextmanager
def server(*flags: str) -> Iterator[None]:
    process = start_max_serve(
        MODEL_ID, "--quantization-encoding", "float32", *flags
    )
    try:
        yield
    finally:
        process.terminate()
        process.wait()


def benchmark(concurrency: int, *count: str) -> dict:
    """Runs ``max benchmark`` and returns its result file. ``count`` is
    ``--num-prompts N`` or ``--num-prompts-multiplier M``."""
    result = WORKDIR / f"result_{len(list(WORKDIR.glob('*.json')))}.json"
    subprocess.run(
        [
            "max",
            "benchmark",
            "--model",
            MODEL_ID,
            "--port",
            "8000",
            "--dataset-name",
            "random",
            "--random-input-len",
            "128",
            "--random-output-len",
            "32",
            "--max-concurrency",
            str(concurrency),
            *count,
            "--result-filename",
            str(result),
            "--log-dir",
            str(WORKDIR / "logs"),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return json.loads(result.read_text())


def summary(run: dict) -> dict[str, float]:
    return {
        "concurrency": run["max_concurrency"],
        "requests/s": run["request_throughput"],
        "output tokens/s": run["total_output_tokens"] / run["duration"],
        "TTFT (ms)": run["mean_ttft_ms"],
        "TPOT (ms)": run["mean_tpot_ms"],
    }


def show(runs: list[dict]) -> None:
    rows = [summary(run) for run in runs]
    print("".join(f"{name:>17}" for name in rows[0]))
    for row in rows:
        print(
            "".join(
                f"{value:>17.1f}"
                if isinstance(value, float)
                else f"{value:>17}"
                for value in row.values()
            )
        )
```

</details>

## MAX's SmolLM2

```python
with server():
    max_runs = [
        benchmark(1, "--num-prompts", "16"),
        benchmark(8, "--num-prompts", "32"),
    ]
show(max_runs)
```

```text
      concurrency       requests/s  output tokens/s        TTFT (ms)        TPOT (ms)
                1              1.5             46.6             85.4             19.4
                8              8.5            271.8            236.0             22.7
```

With 8 requests in flight, the server batches them into each decode step.
Throughput rises several times, while TPOT barely changes: one decode step costs
about the same for 8 sequences as for 1. At this model size, most of a step's
cost is fixed work, such as reading the weights and launching kernels, and
little of it is arithmetic per sequence. The handbook's
[batching](https://handbook.modular.com/inference-optimization/static-dynamic-continuous-batching/)
page explains why. TTFT rises because requests that arrive together are
prefilled together, and a new request waits for the current step to finish.

The concurrency-8 run sends more requests because `max benchmark` doesn't count
all of them. A run this short reports over a trimmed window: at concurrency
above 1 it leaves out the first `max_concurrency` requests sent and the last
`max_concurrency` completed, because the server is still filling up or draining
while they run, and divides by the time from the first counted request's
submission to the last one's completion. With at least 100 requests at
concurrency above 1, it first looks for a steady-state window automatically and
trims only when it finds none. Either way, the result file says which requests
counted (`measured` in each request record).

## Your architecture

Now run the same benchmark against the package from
[Section 7.1](07_01_serve_your_model.md):

```python
with server("--custom-architectures", "minillm.serve", "--no-prefer-module-v3"):
    ours = [
        benchmark(1, "--num-prompts", "8"),
        benchmark(8, "--num-prompts", "24"),
    ]  # fewer: this server is slow
show(ours)
```

```text
      concurrency       requests/s  output tokens/s        TTFT (ms)        TPOT (ms)
                1              0.5             14.7             82.3             67.7
                8              0.2              6.5          17743.1             94.3
```

```python
def gain(runs: list[dict]) -> float:
    return runs[1]["request_throughput"] / runs[0]["request_throughput"]


def gain_all_requests(runs: list[dict]) -> float:
    rates = []
    for run in runs:
        records = run["request_records"]
        start = min(r["submit_time"] for r in records)
        end = max(r["complete_time"] for r in records)
        rates.append(len(records) / (end - start))
    return rates[1] / rates[0]


print(
    f"throughput gain from concurrency 8: MAX's {gain(max_runs):.1f}x, "
    f"yours {gain(ours):.1f}x"
)
print(
    f"over every request, not just the counted ones: "
    f"MAX's {gain_all_requests(max_runs):.1f}x, "
    f"yours {gain_all_requests(ours):.1f}x"
)
assert gain(max_runs) > 1.5 and gain_all_requests(ours) < 1.5
```

```text
throughput gain from concurrency 8: MAX's 5.8x, yours 0.4x
over every request, not just the counted ones: MAX's 5.2x, yours 0.7x
```

<details class="code-fold">
<summary>Show the code <span class="code-fold-lines">32 lines</span></summary>

```python
metrics = ["requests/s", "output tokens/s", "TTFT (ms)", "TPOT (ms)"]
fig = make_subplots(
    rows=1, cols=4, horizontal_spacing=0.07, subplot_titles=metrics
)
for name, runs, color in [
    ("MAX's SmolLM2", max_runs, viz.BLUE),
    ("yours (Section 7.1)", ours, viz.ORANGE),
]:
    rows = [summary(run) for run in runs]
    for col, metric in enumerate(metrics, start=1):
        fig.add_trace(
            go.Bar(
                x=[f"c = {r['concurrency']}" for r in rows],
                y=[r[metric] for r in rows],
                name=name,
                marker_color=color,
                legendgroup=name,
                showlegend=col == 1,
                hovertemplate=f"{name}, %{{x}}: %{{y:,.1f}} {metric}<extra></extra>",
            ),
            row=1,
            col=col,
        )
fig.update_yaxes(type="log", title_text="ms (log scale)", row=1, col=3)
fig.update_layout(
    title="Serving SmolLM2 at concurrency 1 and 8",
    barmode="group",
    height=440,
    legend=dict(y=-0.12, yanchor="top"),
    margin=dict(t=96),
)
viz.show(fig)
```

</details>

<div class="plotly-figure" data-src="generated/07_02_benchmark_serving_10_0.json"></div>

The two servers differ in two ways, and earlier sections explain both:

- TPOT is several times higher. The pipeline model from
  [Section 7.1](07_01_serve_your_model.md) reruns the
  whole sequence for every token, as
  [Section 1.7](01_07_sampling.md) did, so a step
  grows with the sequence. The KV cache from
  [Section 1.8](01_08_kv_cache.md) makes it constant.
- Concurrency doesn't raise throughput. The package declares `max_batch_size` 1,
  so the server runs one request at a time and queues the rest. The queue shows
  up in TTFT: a request waits for the ones ahead of it to finish completely.

The reported throughput even drops, to half the concurrency-1 rate or less,
while counted over every request of the run it drops much less. The trim
explains the difference. Here is every request of your server's concurrency-8
run, from the result file's request records. The shaded span is the window that
`max benchmark` divides by:

<details class="code-fold">
<summary>Show the code <span class="code-fold-lines">56 lines</span></summary>

```python
records = sorted(ours[1]["request_records"], key=lambda r: r["submit_time"])
start = records[0]["submit_time"]
counted = [r for r in records if r["measured"]]
window = (
    min(r["submit_time"] for r in counted) - start,
    max(r["complete_time"] for r in counted) - start,
)
rows = [
    f"request {i}" + ("" if r["measured"] else " (not counted)")
    for i, r in enumerate(records)
]
fig = go.Figure(
    [
        go.Bar(
            y=rows,
            x=[r["ttft_ms"] / 1000 for r in records],
            base=[r["submit_time"] - start for r in records],
            orientation="h",
            name="waiting, then prefill",
            marker_color=viz.MUTED,
            hovertemplate="%{y}: first token after %{x:.1f} s<extra></extra>",
        ),
        go.Bar(
            y=rows,
            x=[(r["latency_ms"] - r["ttft_ms"]) / 1000 for r in records],
            base=[
                r["submit_time"] - start + r["ttft_ms"] / 1000 for r in records
            ],
            orientation="h",
            name="generating",
            marker_color=[
                viz.ORANGE if r["measured"] else viz.GRID for r in records
            ],
            hovertemplate="%{y}: generating for %{x:.1f} s<extra></extra>",
        ),
    ]
)
fig.add_vrect(
    x0=window[0],
    x1=window[1],
    fillcolor=viz.ORANGE,
    opacity=0.08,
    line_width=0,
    annotation_text=(
        f"counted: {len(counted)} requests in {window[1] - window[0]:.0f} s"
    ),
    annotation_position="top left",
)
fig.update_layout(
    title="Your server at concurrency 8: every request of the run",
    barmode="overlay",
    height=560,
    xaxis_title="seconds since the first request was sent",
    yaxis=dict(autorange="reversed", showgrid=False),
)
viz.show(fig)
```

</details>

<div class="plotly-figure" data-src="generated/07_02_benchmark_serving_12_0.json"></div>

The first 8 requests are sent at once and finish one after another. Each
finished request frees a client slot, so the counted requests, 8 to 15, are
sent while 7 earlier requests still wait in the server's queue. Their window
also holds the service time of those 7 uncounted requests: about 15 requests'
work, credited to 8. Over every request, from the first submission to the last
completion, throughput stays close to the concurrency-1 rate, as it should for
a server that runs one request at a time. The trim suits a server that batches,
where the ramp-up and the drain distort the numbers. For a server that
serializes requests, read the untrimmed rate, or pass
`--skip-first-n-requests 0 --skip-last-n-requests 0`.

The engine in [Chapter 5](ch05.md) fixed both with a
paged KV cache and continuous batching. MAX's own architectures get them from
MAX's serving stack, which is why a production architecture registers a
decoder on MAX's paged KV cache, as the second registration in
[Section 7.1](07_01_serve_your_model.md) does.

## Reading benchmarks well

Measure a server at several concurrency levels, not one, and plot latency
against throughput. `--max-concurrency 1,2,4,8,16,32` runs such a sweep in one
command. Pair it with `--num-prompts-multiplier 8` instead of a fixed
`--num-prompts`: each level then sends 8 requests per slot, so the trim never
uses up a level's requests, as a fixed 32 would at concurrency 32. The cell
below runs the levels one at a time with the same flag, against MAX's SmolLM2:

```python
with server():
    benchmark(1, "--num-prompts", "2")  # warm up the new server
    sweep = [
        benchmark(c, "--num-prompts-multiplier", "8")
        for c in [1, 2, 4, 8, 16, 32]
    ]
show(sweep)
```

```text
      concurrency       requests/s  output tokens/s        TTFT (ms)        TPOT (ms)
                1              1.4             45.3            109.9             19.2
                2              2.8             88.0             95.6             20.4
                4              4.8            153.1            148.1             22.2
                8              8.3            265.0            280.8             22.1
               16             10.8            346.3            609.3             24.2
               32             14.2            455.9            999.4             33.3
```

<details class="code-fold">
<summary>Show the code <span class="code-fold-lines">55 lines</span></summary>

```python
TARGET_MS = 500
rows = [summary(run) for run in sweep]
throughput = [r["output tokens/s"] for r in rows]
p95_ttft = [run["p95_ttft_ms"] for run in sweep]
meets = [t <= TARGET_MS for t in p95_ttft]
best = max(
    (i for i, ok in enumerate(meets) if ok),
    key=lambda i: throughput[i],
    default=None,
)
fig = go.Figure(
    go.Scatter(
        x=throughput,
        y=p95_ttft,
        mode="lines+markers+text",
        text=[f"c = {r['concurrency']}" for r in rows],
        textposition="top left",
        marker=dict(
            size=10, color=[viz.BLUE if ok else viz.RED for ok in meets]
        ),
        line=dict(color=viz.MUTED),
        customdata=[
            [r["concurrency"], run["mean_tpot_ms"]]
            for r, run in zip(rows, sweep)
        ],
        hovertemplate="concurrency %{customdata[0]}: %{x:.0f} tokens/s, "
        "p95 TTFT %{y:.0f} ms, mean TPOT %{customdata[1]:.1f} ms<extra></extra>",
    )
)
fig.add_hline(
    y=TARGET_MS,
    line=dict(color=viz.INK_SECONDARY, dash="dash"),
    annotation_text=f"target: p95 TTFT {TARGET_MS} ms",
    annotation_position="top left",
)
if best is not None:
    fig.add_annotation(
        x=throughput[best],
        y=p95_ttft[best],
        text="highest throughput<br>within the target",
        showarrow=True,
        arrowcolor=viz.INK_SECONDARY,
        ax=40,
        ay=40,
    )
fig.update_layout(
    title="MAX's SmolLM2: latency against throughput as concurrency rises",
    height=440,
    showlegend=False,
    xaxis=dict(title="output tokens/s", rangemode="tozero"),
    yaxis=dict(
        title="p95 time to first token (ms, log scale)", type="log", dtick=1
    ),
)
viz.show(fig)
```

</details>

<div class="plotly-figure" data-src="generated/07_02_benchmark_serving_15_0.json"></div>

Each point is one concurrency level, and red points miss a target of a
95th-percentile TTFT under 500 ms. Throughput keeps rising with concurrency, but
TTFT rises faster once the batch fills. In the run on this page, the highest
throughput within the target came at concurrency 8, and concurrency 16 added
about a third more throughput at more than twice the p95 TTFT. The levels
at 16 and 32 send 128 and 256 requests, enough for `max benchmark`'s
steady-state window; the others are trimmed. Your numbers depend on your
hardware.

- Benchmark the workload you will serve. Input and output lengths change
  the results a lot: long prompts stress prefill and TTFT, and long answers
  stress decode and TPOT.
- Sweep concurrency, as above, with `--num-prompts-multiplier` rather than a
  fixed `--num-prompts`. Throughput grows ever more slowly once the hardware is
  busy, while latency keeps rising.
- Set a latency target first, such as a TTFT under 500 ms at the 95th
  percentile, then find the highest throughput that meets it. The result
  file stores percentiles (`p95_ttft_ms`, `p99_tpot_ms`, and so on) next to
  the means, and `--goodput ttft:500 tpot:50` adds the share of requests that
  met every target (`goodput_attainment`).

The handbook's
[LLM performance benchmarks](https://handbook.modular.com/inference-optimization/llm-performance-benchmarks/)
page covers benchmark design, and `max benchmark --help` lists the other
datasets and options.

## Exercises

1. Extend the sweep of MAX's SmolLM2 to concurrency 64 and 128 with
   `--num-prompts-multiplier 4` and `--goodput ttft:500`. Where does
   throughput stop growing, and at which level does `goodput_attainment` fall
   below 90%?
2. Repeat the MAX run with `--random-input-len 1024`. Which metric changes most,
   and why?
3. Do the second exercise of
   [Section 7.1](07_01_serve_your_model.md), raise
   `max_batch_size` in its package to 4, and benchmark it again at concurrency
   8.

## Next

[Section 7.3](07_03_structured_output.md) constrains
what the server generates: JSON that follows a schema, and calls to your own
functions.

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

```python
import json
import subprocess
import tempfile
import time
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from minillm import viz
from minillm.config import MODEL_ID

WORKDIR = Path(tempfile.mkdtemp())
```

## What to measure

Four numbers describe a serving run. The handbook's
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
conversations with varied lengths. The two helpers below start a server and
run a benchmark from Python, then read the result file:

```python
@contextmanager
def server(*flags: str) -> Iterator[None]:
    process = subprocess.Popen(
        [
            "max",
            "serve",
            "--model",
            MODEL_ID,
            "--port",
            "8000",
            "--quantization-encoding",
            "float32",
            *flags,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        while True:
            try:
                urllib.request.urlopen("http://localhost:8000/health")
                break
            except OSError:
                if process.poll() is not None:
                    raise RuntimeError(
                        "max serve exited; run it in a terminal to see why"
                    )
                time.sleep(1)
        yield
    finally:
        process.terminate()
        process.wait()


def benchmark(concurrency: int, prompts: int) -> dict[str, float]:
    result = WORKDIR / f"result_{concurrency}.json"
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
            "--num-prompts",
            str(prompts),
            "--max-concurrency",
            str(concurrency),
            "--result-filename",
            str(result),
            "--log-dir",
            str(WORKDIR / "logs"),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    run = json.loads(result.read_text())
    return {
        "concurrency": concurrency,
        "requests/s": run["request_throughput"],
        "output tokens/s": run["total_output_tokens"] / run["duration"],
        "TTFT (ms)": run["mean_ttft_ms"],
        "TPOT (ms)": run["mean_tpot_ms"],
    }


def show(rows: list[dict[str, float]]) -> None:
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

At concurrency 8, `max benchmark` leaves the first and last 8 requests out
of its statistics, because the server is still filling up or draining while
they run. The runs at concurrency 8 therefore send more requests.

## MAX's SmolLM2

```python
with server():
    max_runs = [benchmark(1, prompts=16), benchmark(8, prompts=32)]
show(max_runs)
```

```text
      concurrency       requests/s  output tokens/s        TTFT (ms)        TPOT (ms)
                1              1.4             45.2             80.7             20.2
                8              8.6            275.6            275.2             21.1
```

With 8 requests in flight, the server batches them into each decode step.
Throughput rises several times, while TPOT barely changes: one decode step costs
about the same for 8 sequences as for 1. At this model size, most of a step's
cost is fixed work, such as reading the weights and launching kernels, and
little of it is arithmetic per sequence. The handbook's
[batching](https://handbook.modular.com/inference-optimization/static-dynamic-continuous-batching/)
page explains why. TTFT rises because requests that arrive together are
prefilled together, and a new request waits for the current step to finish.

## Your architecture

Now run the same benchmark against the package from
[Section 7.1](07_01_serve_your_model.md):

```python
with server("--custom-architectures", "minillm.serve", "--no-prefer-module-v3"):
    ours = [
        benchmark(1, prompts=8),
        benchmark(8, prompts=24),
    ]  # fewer: this server is slow
show(ours)
```

```text
      concurrency       requests/s  output tokens/s        TTFT (ms)        TPOT (ms)
                1              0.5             14.4             73.5             69.1
                8              0.2              6.3          18269.0             96.8
```

```python
max_gain = max_runs[1]["requests/s"] / max_runs[0]["requests/s"]
ours_gain = ours[1]["requests/s"] / ours[0]["requests/s"]
print(
    f"throughput gain from concurrency 8: MAX's {max_gain:.1f}x, yours {ours_gain:.1f}x"
)
assert max_gain > 1.5 and ours_gain < 1.5

metrics = ["requests/s", "output tokens/s", "TTFT (ms)", "TPOT (ms)"]
fig = make_subplots(
    rows=1, cols=4, horizontal_spacing=0.07, subplot_titles=metrics
)
for name, rows, color in [
    ("MAX's SmolLM2", max_runs, viz.BLUE),
    ("yours (Section 7.1)", ours, viz.ORANGE),
]:
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
fig.update_layout(
    title="Serving SmolLM2 at concurrency 1 and 8",
    barmode="group",
    height=440,
    legend=dict(y=-0.12, yanchor="top"),
    margin=dict(t=96),
)
viz.show(fig)
```

```text
throughput gain from concurrency 8: MAX's 6.1x, yours 0.4x
```

<div class="plotly-figure" data-src="generated/07_02_benchmark_serving_9_1.json"></div>

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

The engine in [Chapter 5](ch05.md) fixed both with a
paged KV cache and continuous batching. MAX's own architectures get them from
MAX's serving stack, which is why a production architecture implements a decoder
on MAX's paged KV cache
([Section 7.1](07_01_serve_your_model.md) links the
guide).

## Reading benchmarks well

- Benchmark the workload you will serve. Input and output lengths change
  the results a lot: long prompts stress prefill and TTFT, and long answers
  stress decode and TPOT.
- Sweep concurrency, for example `--max-concurrency 1,4,16,64`, and plot
  throughput against TTFT or TPOT. Throughput levels off once the GPU is
  busy, while latency keeps rising. The useful operating point is just
  before throughput levels off.
- Set a latency target first, such as a TTFT under 500 ms at the 95th
  percentile, then find the highest throughput that meets it. The result
  file stores percentiles (`p95_ttft_ms`, `p99_tpot_ms`, and so on) next to
  the means.

The handbook's
[LLM performance benchmarks](https://handbook.modular.com/inference-optimization/llm-performance-benchmarks/)
page covers benchmark design, and `max benchmark --help` lists the other
datasets and options.

## Exercises

1. Sweep the concurrency of MAX's SmolLM2 over 1, 2, 4, 8, 16, and 32. Where
   does throughput stop growing, and what happens to TTFT after that?
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

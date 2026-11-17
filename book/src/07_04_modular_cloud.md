# 7.4 Modular's cloud

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/modular/max-llm-book/blob/main/notebooks/07_04_modular_cloud.ipynb)
[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/modular/max-llm-book/blob/main/notebooks/07_04_modular_cloud.ipynb)
[![View on GitHub](https://img.shields.io/badge/GitHub-notebook-black?logo=github)](https://github.com/modular/max-llm-book/blob/main/notebooks/07_04_modular_cloud.ipynb)

Everything in this book ran on your own machine: models of up to 0.6B
parameters, with multi-device meshes simulated on a CPU. Serving
state-of-the-art open models to real traffic takes more: the largest models span
many GPUs, capacity has to follow demand, and someone has to respond when a node
fails. [Modular's cloud](https://www.modular.com/) is an inference service that
runs those models on MAX, the framework you used in this book. This last section
points your client code at it by changing the base URL, the API key, and the
model name.

```python
import os

from openai import OpenAI

from minillm.config import MODEL_ID
from minillm.server import start_max_serve
```

## Two ways to call a hosted model

Modular's cloud serves popular open models behind the same OpenAI-compatible
API that the earlier sections of this chapter used:

- [Shared endpoints](https://www.modular.com/inference/shared-endpoints)
  are billed per token, and their capacity scales automatically, down to
  zero when idle.
- [Dedicated endpoints](https://www.modular.com/inference/dedicated-endpoints)
  reserve GPUs for you and are billed per minute, for consistent latency and
  guaranteed throughput.

The [pricing](https://www.modular.com/pricing) page compares them.

## The same client, a different URL

Every server in [Chapter 7](ch07.md) spoke the OpenAI
API, and so do Modular's endpoints. Here is the client from
[Section 7.1](07_01_serve_your_model.md) against a
local server again:

```python
server = start_max_serve(MODEL_ID, "--quantization-encoding", "float32")


def ask(client: OpenAI, model: str, question: str) -> str:
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": question}],
        max_tokens=60,
        temperature=0,
    )
    return completion.choices[0].message.content


local = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
print(ask(local, MODEL_ID, "What is the capital of France?"))
server.terminate()
_ = server.wait()
```

```text
The capital of France is Paris.
```

To call a hosted model instead:

1. Sign up at [console.modular.com](https://console.modular.com/signup).
2. Create an API key with model access on the
   [API keys](https://console.modular.com/api_keys) page, and set it as the
   `MODULAR_API_KEY` environment variable. In Colab, add it as a secret and
   copy it into `os.environ`.
3. Pick a model on the [endpoints](https://console.modular.com/endpoints)
   page, and set its model ID as `MODULAR_MODEL`.

Only the base URL, the key, and the model name change. The cell skips the
request when no key is set:

```python
api_key = os.environ.get("MODULAR_API_KEY")
if api_key is None:
    print("Set MODULAR_API_KEY and MODULAR_MODEL to call a hosted model.")
else:
    hosted = OpenAI(base_url="https://api.modular.com/v1", api_key=api_key)
    print(
        ask(
            hosted,
            os.environ["MODULAR_MODEL"],
            "What is the capital of France?",
        )
    )
```

```text
Set MODULAR_API_KEY and MODULAR_MODEL to call a hosted model.
```

The API is the same, so code written against it, such as the structured output
in [Section 7.3](07_03_structured_output.md), works
with hosted models that support the feature. `max benchmark` measures a hosted
endpoint as it measured your local server. The `mcloud` backend sends the key
from `OPENAI_API_KEY`. A hosted endpoint manages its own cache and doesn't
expose server metrics, so the command skips both. If the model ID isn't a
Hugging Face repo, add `--tokenizer` with the repo of its tokenizer:

```bash
OPENAI_API_KEY=$MODULAR_API_KEY max benchmark --backend mcloud \
  --base-url https://api.modular.com --endpoint /v1/chat/completions \
  --no-flush-prefix-cache --no-collect-server-stats \
  --model $MODULAR_MODEL \
  --dataset-name random --random-input-len 128 --random-output-len 32 \
  --num-prompts 32 --max-concurrency 8
```

Before you choose between shared and dedicated endpoints, benchmark with your
real prompt and answer lengths; the latency targets from
[Section 7.2](07_02_benchmark_serving.md) apply
unchanged. The handbook's
[choosing the right model](https://handbook.modular.com/getting-started/choosing-the-right-model/)
and
[calculating GPU memory](https://handbook.modular.com/getting-started/calculating-gpu-memory-for-llms/)
pages help you choose a model and estimate the GPUs it needs.

## What you built

You started this book with three lines of `AutoPipeline`. Since then you:

- wrote SmolLM2 from tensors up, checked it against MAX's implementation, and
  added sampling and a KV cache
  ([Chapter 1](ch01.md));
- sharded it across a device mesh
  ([Chapter 2](ch02.md));
- wrote a Mojo kernel and built a graph by hand
  ([Chapter 3](ch03.md));
- tested, benchmarked, and profiled it
  ([Chapter 4](ch04.md));
- built a serving engine with continuous batching, a paged KV cache, prefix
  caching, and speculative decoding
  ([Chapter 5](ch05.md));
- brought up Qwen3, Gemma 3, and a mixture of experts
  ([Chapter 6](ch06.md));
- served your own architecture, measured it, and constrained its output
  ([Chapter 7](ch07.md)).

A model takes the same path into MAX, at a larger size. To keep going:

- Read MAX's guide to
  [model bring-up](https://max.modular.com/develop/model-bringup-workflow) and
  port a model you care about.
- Work through the
  [Mojo GPU puzzles](https://github.com/modular/mojo-gpu-puzzles) to write the
  kernels behind it.
- Read the [handbook](https://handbook.modular.com/) for the ideas behind every
  chapter.
- [Sign up](https://console.modular.com/signup) to run state-of-the-art open
  models on Modular's cloud, or
  [request a demo](https://www.modular.com/request-demo) to talk about your
  workload.

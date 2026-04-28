# Run the model

Before building GPT-2 from scratch, run it. The `max serve` command exposes an
OpenAI-compatible HTTP API for models you run through it, including this
tutorial's custom GPT-2. That differs from wiring PyTorch or Hugging Face
Transformers for inference and HTTP serving yourself: you add a small
architecture package and get a live endpoint without stitching together serving,
compilation, and weight loading by hand.

You'll see text generation working in minutes; then the build chapters
explain every component that makes it work.

## Start the server

Start the server with:

```sh
pixi run serve
```

That command runs:

```sh
max serve --custom-architectures gpt2_arch --model gpt2
```

On the first run, MAX downloads the pretrained GPT-2 weights from Hugging Face
(≈ 548 MB) and compiles the model. Your first run might take a minute or two;
later runs use cached weights and start faster. When the server is ready you'll
see:

```text
Server ready on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

## Query the model

GPT-2 is a *completion* model, not a chat model. It continues text rather than
answering questions: pass it the start of a sentence and it generates what
comes next. Use the `/v1/completions` endpoint with a `prompt` field:

```sh
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt2",
    "prompt": "In the beginning",
    "max_tokens": 30,
    "temperature": 0
  }'
```

`temperature: 0` picks the highest-probability token at each step, producing
deterministic output. Try values between 0.7 and 1.0 for more varied
completions. Or query with the Python `openai` client (requires
`pip install openai`):

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

response = client.completions.create(
    model="gpt2",
    prompt="In the beginning",
    max_tokens=30,
    temperature=0,
)
print(response.choices[0].text)
```

The completion text is in `response.choices[0].text`.

## How it works

`gpt2_arch/` is a custom architecture package that implements the interface
`max serve` expects. When you send a request, `max serve` tokenizes the prompt,
runs the token IDs through the compiled model graph, and samples the next token
from the output logits. It repeats that until `max_tokens` is reached, then
returns the detokenized completion.

```text
gpt2_arch/
├── __init__.py        # registers the architecture with max serve
├── arch.py            # declares the supported model name and config
├── model_config.py    # KV cache params, max sequence length
├── gpt2.py            # the model architecture you build in this tutorial
├── model.py           # loads weights, compiles, and serves the model
└── weight_adapters.py # adapts GPT-2 Conv1D weights to MAX format
```

## What's next

The next sections build the GPT-2 architecture and serving infrastructure from
scratch, component by component: the model definition, weight loading, and the
package that connects everything to `max serve`. Start with
[Model configuration](./step_01.md).

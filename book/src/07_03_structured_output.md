# 7.3 Structured output and tools

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/modular/max-llm-book/blob/main/notebooks/07_03_structured_output.ipynb)
[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/modular/max-llm-book/blob/main/notebooks/07_03_structured_output.ipynb)
[![View on GitHub](https://img.shields.io/badge/GitHub-notebook-black?logo=github)](https://github.com/modular/max-llm-book/blob/main/notebooks/07_03_structured_output.ipynb)

A chat answer is for people. A program needs output it can parse: a JSON
object with the right fields, or the name of a function to call with its
arguments. Instructions in the prompt don't get that reliably, least of all
from a small model. This section makes the server *guarantee* the format
with structured output, then builds function calling on top of it.

```python
import json
import subprocess
import time
import urllib.request

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

from minillm.config import MODEL_ID
```

Structured output is off by default, and `--enable-structured-output`
turns it on:

```python
server = subprocess.Popen(
    [
        "max",
        "serve",
        "--model",
        MODEL_ID,
        "--port",
        "8000",
        "--quantization-encoding",
        "float32",
        "--enable-structured-output",
    ],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
while True:
    try:
        urllib.request.urlopen("http://localhost:8000/health")
        break
    except OSError:
        if server.poll() is not None:
            raise RuntimeError(
                "max serve exited; run it in a terminal to see why"
            )
        time.sleep(1)
client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")


def chat(messages: list[dict[str, str]], **options: object) -> str:
    completion = client.chat.completions.create(
        model=MODEL_ID,
        messages=messages,
        max_tokens=100,
        temperature=0,
        **options,
    )
    return completion.choices[0].message.content
```

## JSON that follows a schema

Without a constraint, the model answers in prose:

```python
question = [
    {
        "role": "user",
        "content": "Tell me about Rome: its country and three landmarks.",
    }
]
print(chat(question))
```

```text
Rome, the Eternal City, is a city that has witnessed so much history and transformation. As a helpful AI assistant, I'm here to share with you the fascinating facts about Rome, its country, and three iconic landmarks.

Rome, the Eternal City, is a city that has witnessed so much history and transformation. It's a place where ancient civilizations like the Romans, Greeks, and Etruscans have left their mark, and it's a place where the Romans have built a lasting
```

Describe the answer you want as a Pydantic model instead. Pydantic turns it
into a JSON schema, which the request passes as `response_format`:

```python
class City(BaseModel):
    model_config = ConfigDict(extra="forbid")  # no fields beyond these
    name: str
    country: str
    landmarks: list[str] = Field(min_length=3, max_length=3)


schema = {
    "type": "json_schema",
    "json_schema": {"name": "City", "schema": City.model_json_schema()},
}
answer = chat(question, response_format=schema)
print(answer)
city = City.model_validate_json(answer)
city
```

```text
{"name":"Rome","country":"Italy","landmarks":["The Colosseum","The Roman Forum","The Pantheon"]}
City(name='Rome', country='Italy', landmarks=['The Colosseum', 'The Roman Forum', 'The Pantheon'])
```

`model_validate_json` raises if the text doesn't match the schema. With
structured output, the text always matches.

## How the server guarantees it

The guarantee comes from the sampler. MAX compiles the schema into a grammar,
and at every decode step the grammar says which tokens may come next: after
`{"name": "Rome", `, only tokens that continue toward `"country"`. MAX masks
every other token out of the logits before sampling, as the top-p filter in
[Section 1.7](01_07_sampling.md) did with \\(-\\infty\\),
so the model can only pick a token that keeps the output valid. Generation ends
when the grammar is complete.

The schema fixes the *form* of the answer and leaves the *facts* to the model.
The model can skip any requirement that the schema leaves out: without
`min_length`, a small model may return an empty `landmarks` list. Put every
requirement you can into the schema, and check the values that matter. The
handbook's
[structured outputs](https://handbook.modular.com/model-interaction/structured-outputs/)
page covers schema design and the cost of constrained decoding.

## Function calling

A *tool* is a function that the model may ask your program to run: look up
the weather, query a database, call an API. The model never runs anything
itself. It writes the function's name and arguments, your code runs the
function, and the result goes back into the conversation. A tool call is
therefore a structured output whose schema is "one of these functions, with
valid arguments for it".

Here are two tools, each a Python function plus a schema for its
arguments:

```python
RATES = {"USD": 1.0, "EUR": 0.92, "JPY": 151.0}


def get_weather(city: str) -> dict[str, object]:
    return {
        "city": city,
        "temperature_c": 18,
        "sky": "light rain",
    }  # a stand-in for a weather API


def convert_currency(
    amount: float, source: str, target: str
) -> dict[str, object]:
    return {
        "amount": round(amount / RATES[source] * RATES[target], 2),
        "currency": target,
    }


def arguments(**properties: dict[str, object]) -> dict[str, object]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


TOOLS = {
    "get_weather": (
        get_weather,
        "Gets the current weather in a city.",
        arguments(city={"type": "string"}),
    ),
    "convert_currency": (
        convert_currency,
        "Converts an amount between currencies.",
        arguments(
            amount={"type": "number"},
            source={"enum": list(RATES)},
            target={"enum": list(RATES)},
        ),
    ),
}
```

The schema for a call allows any one of the tools: `anyOf` lists one object
per tool, each with its fixed `name` and its own arguments. `enum` limits
the currencies to the ones `RATES` knows, so the model can't ask for one the
function would fail on. The system prompt tells the model what the tools do:

```python
call_schema = {
    "anyOf": [
        {
            "type": "object",
            "properties": {"name": {"const": name}, "arguments": args},
            "required": ["name", "arguments"],
            "additionalProperties": False,
        }
        for name, (_, _, args) in TOOLS.items()
    ]
}
system = (
    "You can call these functions. Reply with one call as JSON.\n"
    + "\n".join(
        f"- {name}: {description} Arguments: {json.dumps(args['properties'])}"
        for name, (_, description, args) in TOOLS.items()
    )
)


def call_tool(question: str) -> tuple[dict[str, object], dict[str, object]]:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]
    call = json.loads(
        chat(
            messages,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "call", "schema": call_schema},
            },
        )
    )
    function = TOOLS[call["name"]][0]
    return call, function(**call["arguments"])


for question in [
    "What's the weather in Tokyo?",
    "How much is 20 US dollars in euros?",
]:
    call, result = call_tool(question)
    print(f"{question}\n  call:   {call}\n  result: {result}")
```

```text
What's the weather in Tokyo?
  call:   {'name': 'get_weather', 'arguments': {'city': 'Tokyo'}}
  result: {'city': 'Tokyo', 'temperature_c': 18, 'sky': 'light rain'}
How much is 20 US dollars in euros?
  call:   {'name': 'convert_currency', 'arguments': {'amount': 20, 'source': 'USD', 'target': 'EUR'}}
  result: {'amount': 18.4, 'currency': 'EUR'}
```

Every call parses and runs, since the grammar allows nothing else. Whether
the model picks the *right* tool is up to the model; this one does on these
questions.

The last step gives the result back so the model can answer. Here the
result goes into the system prompt:

```python
question = "Should I take an umbrella in Paris today?"
call, result = call_tool(question)
print("call:", call)
print(
    chat(
        [
            {
                "role": "system",
                "content": f"Current weather: {json.dumps(result)}",
            },
            {"role": "user", "content": question},
        ]
    )
)
```

```text
call: {'name': 'get_weather', 'arguments': {'city': 'Paris'}}
As a traveler, it's essential to consider the weather forecast and potential weather-related issues before making a decision. Paris is a city known for its rain, and the city's weather can be unpredictable.

If you're planning to visit Paris in the morning, it's best to take an umbrella. The city's rain can be heavy and unpredictable, especially in the mornings. A rain umbrella can provide some protection from the rain, but it's not a guarantee.

If you're
```

```python
server.terminate()
_ = server.wait()
```

A 135M-parameter model uses the result loosely. Larger models follow it
much more closely, especially ones trained for function calling.

Those models also take tools through the `tools` parameter of the OpenAI API.
Their chat templates describe the tools to the model in the format it was
trained on, and the model writes calls in its own format. MAX parses those calls
with a *tool parser* for each model family, such as Llama, Qwen, and DeepSeek,
and returns them as `message.tool_calls`; `tool_choice="required"` forces a
call. SmolLM2's chat template has no place for tools, which is why this section
built the protocol by hand. The mechanism is the same in both cases: the model
writes a structured call, and your code runs it. The handbook's
[function calling](https://handbook.modular.com/model-interaction/function-calling/)
page covers the API and the agent loops built on it.

## Exercises

1. Remove `Field(min_length=3, max_length=3)` from `City`. What does the model
   return now?
2. Add a third tool, `get_time(city)`, and a question that needs it. Does the
   model choose it?
3. `max benchmark` takes `--response-format` with a JSON schema. Benchmark the
   server with and without one, as in
   [Section 7.2](07_02_benchmark_serving.md). How much
   does constrained decoding cost per token?

## Next

[Section 7.4](07_04_modular_cloud.md) points the same
client at state-of-the-art open models on Modular's cloud.

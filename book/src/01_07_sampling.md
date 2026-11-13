# 1.7 Generation and sampling

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/modular/max-llm-book/blob/main/notebooks/01_07_sampling.ipynb)
[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/modular/max-llm-book/blob/main/notebooks/01_07_sampling.ipynb)
[![View on GitHub](https://img.shields.io/badge/GitHub-notebook-black?logo=github)](https://github.com/modular/max-llm-book/blob/main/notebooks/01_07_sampling.ipynb)

Your model turns tokens into next-token logits. This section turns logits into
text: the generation loop, and the sampling strategies behind the `temperature`,
`top_k`, and `top_p` settings from
[Section 1.1](01_01_run_and_serve.md). Everything runs
on your own model from
[Section 1.6](01_06_full_model.md).

```python
import logging
import time
import warnings

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from max import functional as F
from max.driver import CPU
from max.dtype import DType
from max.functional import random
from max.pipelines import AutoModel, AutoTokenizer
from max.tensor import Tensor
from max.tensor.tensor import TensorType

from minillm import viz
from minillm.device import pick_device
from minillm.config import MODEL_ID, LlamaConfig
from minillm.model import load_model
from minillm.weights import load_checkpoint

logging.getLogger("max").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="The eager interpreter failed")
DEVICE = pick_device()  # a GPU if MAX finds one, otherwise the CPU

config = LlamaConfig.from_pretrained()
tokenizer = AutoTokenizer(MODEL_ID)
model = load_model(config, load_checkpoint(), DEVICE)
compiled = model.compile(
    TensorType(DType.int64, ("batch", "seq_len"), device=DEVICE)
)
```

The generation loop needs one thing from the model: the logits for the next
token, given the tokens so far.

```python
def next_logits(ids: list[int]) -> Tensor:
    logits = compiled(Tensor(np.array([ids]), device=DEVICE))
    return logits[0, -1]  # [vocab_size]
```

## Greedy decoding

The simplest strategy picks the most likely token every time:

```python
prompt = tokenizer(["The capital of France is"])["input_ids"][0]
ids = list(prompt)
for _ in range(24):
    ids.append(int(F.argmax(next_logits(ids)).item()))
print(tokenizer.decode(ids))
```

```text
The capital of France is Paris. Paris is the largest city in France and the capital of the French department of the Espace de Paris.
```

Greedy decoding is deterministic, so it is the one strategy you can compare
token for token with MAX's own pipeline. `top_k=1` makes `AutoModel` greedy
too:

```python
reference = AutoModel(MODEL_ID, quantization_encoding="float32")
expected = reference.generate(
    input_ids=[prompt], max_new_tokens=24, top_k=1
).sequences[0]
reference.close()
print("same tokens as AutoModel:", list(expected[-24:]) == ids[-24:])
assert list(expected[-24:]) == ids[-24:]
```

```text
same tokens as AutoModel: True
```

Greedy text is coherent but tends to repeat itself and to pick the blandest
continuation. Sampling picks tokens at random, in proportion to their
probability, to get more varied text.

## Temperature

Softmax turns logits \\(z\\) into probabilities. Dividing the logits by a
*temperature* \\(T\\) first reshapes the distribution:

\\[p\_i = \\frac{e^{z\_i / T}}{\\sum\_j e^{z\_j / T}}\\]

Below 1, the likely tokens get likelier; above 1, the distribution flattens
toward uniform. As \\(T \\to 0\\), sampling becomes greedy. Move the slider to
see it on the twelve likeliest next tokens of a story:

```python
story = tokenizer(["Once upon a time, there was a"])["input_ids"][0]
logits = next_logits(story)
top = np.argsort(logits.to_numpy())[::-1][:12]
labels = [repr(tokenizer.decode([int(i)])) for i in top]

temperatures = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
steps = []
for temperature in temperatures:
    probs = F.softmax(logits / temperature).to_numpy()
    steps.append(
        [
            go.Bar(
                x=labels,
                y=probs[top],
                marker_color=viz.BLUE,
                hovertemplate="%{x}: %{y:.3f}<extra></extra>",
            )
        ]
    )
fig = go.Figure(
    layout=dict(
        title="Next-token probabilities after 'Once upon a time, there was a'",
        yaxis=dict(title="probability", range=[0, 1]),
        height=460,
        bargap=0.25,
    )
)
viz.show(
    viz.with_slider(
        fig, steps, [str(t) for t in temperatures], "temperature ", active=4
    )
)
```

<div class="plotly-figure" data-src="generated/01_07_sampling_10_0.json"></div>

## Sampling with MAX operations

To sample from a distribution, you could compute the softmax and its
cumulative sum, then search it with a random number. The *Gumbel-max trick*
does it with operations you already have: add Gumbel noise
\\(-\\log(-\\log u)\\), with \\(u\\) uniform in \\((0, 1)\\), to the scaled logits, and
take the argmax:

\\[\\text{token} = \\arg\\max\_i \\left( \\frac{z\_i}{T} - \\log(-\\log u\_i) \\right),
\\qquad u\_i \\sim \\mathrm{Uniform}(0, 1)\\]

The sampled tokens follow the softmax distribution exactly, without
computing the softmax. A quick check against the probabilities:

```python
def gumbel_argmax(scores: Tensor) -> Tensor:
    u = random.uniform(
        scores.shape,
        range=(1e-6, 1.0),
        dtype=scores.dtype,
        device=scores.device,
    )
    return F.argmax(scores - F.log(-F.log(u)))


toy = Tensor(np.array([2.0, 1.0, 0.5, 0.1, -1.0], np.float32), device=DEVICE)
counts = np.bincount(
    [int(gumbel_argmax(toy).item()) for _ in range(2000)], minlength=5
)
print("sampled:", (counts / 2000).round(3))
print("softmax:", F.softmax(toy).to_numpy().round(3))
assert np.abs(counts / 2000 - F.softmax(toy).to_numpy()).max() < 0.05
```

```text
sampled: [0.556 0.205 0.133 0.08  0.026]
softmax: [0.559 0.205 0.125 0.084 0.028]
```

## Top-k and top-p

Even at a sensible temperature, the long tail of unlikely tokens adds up,
and one bad draw can derail the text. Two filters cut the tail before
sampling:

- Top-k keeps the \\(k\\) most likely tokens.
- Top-p (nucleus sampling) keeps the smallest set of tokens whose
  probabilities add up to at least \\(p\\), so it keeps few tokens when the
  model is confident and many when it isn't.

Both filters work on tokens sorted from most to least likely. `F.top_k`
returns the \\(k\\) largest scores already sorted, with their token IDs. For
top-p alone, `F.argsort` sorts the whole vocabulary and `F.gather` puts the
scores in that order. `F.cumsum` over the sorted probabilities then gives
the running total, and `F.where` drops every token that the total had
already passed \\(p\\) before reaching. The filtered scores go through the
Gumbel-max trick, and the ID map turns the winning position back into a
token. Without a filter, nothing needs sorting.

```python
from __future__ import annotations

from collections.abc import Callable

from max import functional as F
from max.functional import random
from max.tensor import Tensor


def sample(
    logits: Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> int:
    """Picks a token ID from 1-D ``logits``; ``temperature=0`` is greedy."""
    if temperature == 0:
        return int(F.argmax(logits).item())
    scores, token_ids = logits / temperature, None
    if top_k is not None:
        scores, token_ids = F.top_k(scores, top_k)  # sorted, largest first
    if top_p is not None:
        if token_ids is None:
            token_ids = F.argsort(scores, ascending=False)
            scores = F.gather(scores, token_ids, axis=0)
        probs = F.softmax(scores)
        before = F.cumsum(probs, axis=-1) - probs  # mass of likelier tokens
        scores = F.where(before < top_p, scores, float("-inf"))
    u = random.uniform(
        scores.shape,
        range=(1e-6, 1.0),
        dtype=scores.dtype,
        device=scores.device,
    )
    choice = int(F.argmax(scores - F.log(-F.log(u))).item())
    return choice if token_ids is None else int(token_ids.to_numpy()[choice])


def generate(
    next_logits: Callable[[list[int]], Tensor],
    prompt: list[int],
    max_new_tokens: int,
    eos_token_id: int | None = None,
    **sampling: float | None,
) -> list[int]:
    """Generates up to ``max_new_tokens`` tokens after ``prompt``.

    ``next_logits`` maps the tokens so far to the next token's logits.
    Generation stops early at ``eos_token_id``. Returns only the new tokens.
    """
    ids = list(prompt)
    for _ in range(max_new_tokens):
        token = sample(next_logits(ids), **sampling)
        ids.append(token)
        if token == eos_token_id:
            break
    return ids[len(prompt) :]
```

<small>Saved to `minillm/sampling.py`.</small>

Top-p adapts to the model's confidence. After "The capital of France is", two
tokens hold about three quarters of the probability; after "Once upon a time,
there was a", it spreads over many. Move \\(p\\) and compare how many tokens each
prompt keeps (blue) and cuts (gray). The bars show the likeliest 40; the count
covers the whole vocabulary:

```python
prompts = ["The capital of France is", "Once upon a time, there was a"]
shown = 40
sorted_probs = []
for text in prompts:
    probs = F.softmax(next_logits(tokenizer([text])["input_ids"][0])).to_numpy()
    sorted_probs.append(np.sort(probs)[::-1])

ps = [0.5, 0.75, 0.9, 0.95, 0.99]
rank = np.arange(1, shown + 1)
steps = []
for p in ps:
    group = []
    for axis, probs in zip(["", "2"], sorted_probs):
        before = np.cumsum(probs) - probs  # the same test as `sample`
        kept = before < p
        count = int(kept.sum())
        kept, top = kept[:shown], probs[:shown]
        for mask, color, name in [
            (kept, viz.BLUE, "kept"),
            (~kept, viz.MUTED, "cut"),
        ]:
            group.append(
                go.Bar(
                    x=rank[mask],
                    y=top[mask],
                    marker_color=color,
                    name=name,
                    showlegend=False,
                    xaxis=f"x{axis}",
                    yaxis=f"y{axis}",
                    hovertemplate="rank %{x}: %{y:.3f}<extra>"
                    + name
                    + "</extra>",
                )
            )
        group.append(
            go.Scatter(
                x=[shown],
                y=[0.9],
                mode="text",
                text=[f"{count:,} kept"],
                textposition="middle left",
                showlegend=False,
                hoverinfo="skip",
                xaxis=f"x{axis}",
                yaxis=f"y{axis}",
            )
        )
    steps.append(group)

fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.12,
    subplot_titles=[repr(text) for text in prompts],
)
fig.update_layout(
    height=560,
    bargap=0.15,
    title="The likeliest 40 next tokens, sorted, at temperature 1",
)
fig.update_yaxes(range=[0, 1], title="probability")
fig.update_xaxes(title="rank", row=2, col=1)
viz.show(viz.with_slider(fig, steps, [str(p) for p in ps], "top_p ", active=2))
```

<div class="plotly-figure" data-src="generated/01_07_sampling_16_0.json"></div>

`generate` stops at the model's end-of-sequence token, which a chat model
emits when its answer is complete. Try a few settings. `random.set_seed`
makes a sampled run repeatable:

```python
for settings in [
    dict(temperature=0),
    dict(temperature=0.7, top_p=0.9),
    dict(temperature=1.0, top_k=50),
    dict(temperature=1.5),
]:
    random.set_seed(0)
    new = generate(next_logits, story, 24, config.eos_token_id, **settings)
    print(f"{settings}\n  {tokenizer.decode(new)!r}\n")
```

```text
{'temperature': 0}
  ' young woman named Sarah who was struggling to find her place in the world. She had always been a bit of a rebel'

{'temperature': 0.7, 'top_p': 0.9}
  ' young girl named Lily who was determined to be a better version of herself. She had always been the "perfect" girl'

{'temperature': 1.0, 'top_k': 50}
  ' young musician who had spent their life playing and composing. He was known as "The Weaver" due to his exceptional skill'

{'temperature': 1.5}
  ' neighboring realm on your planet, mist. Even then; Unlocking beastellandsis deathshire istsubbralgenaki'
```

Higher temperatures read less predictably. At 1.5 without a filter, the text
turns to gibberish within a few words. Many chat applications use settings near
`temperature=0.7` and `top_p=0.9`. The handbook's
[inference parameters](https://handbook.modular.com/model-interaction/inference-parameters/)
page covers these and the other settings servers accept, such as stop sequences
and repetition penalties.

## The cost of each token

Generating the token after \\(n\\) tokens runs the model over all \\(n\\) of them.
Time that step at several lengths, with random token IDs since only the
length matters:

```python
rng = np.random.default_rng(0)
lengths = [16, 64, 256, 512, 1024, 2048]
step_ms = []
for length in lengths:
    ids = rng.integers(0, config.vocab_size, length).tolist()
    next_logits(ids).to_numpy()  # warm up this length
    times = []
    for _ in range(3):
        start = time.perf_counter()
        next_logits(ids).to_numpy()
        times.append((time.perf_counter() - start) * 1000)
    step_ms.append(min(times))
    print(f"{length:5} tokens: {step_ms[-1]:6.1f} ms per new token")

fig = go.Figure(
    go.Scatter(
        x=lengths,
        y=step_ms,
        mode="lines+markers",
        line_color=viz.ORANGE,
        hovertemplate="%{x} tokens: %{y:.1f} ms<extra></extra>",
    )
)
fig.update_layout(
    title="Time to generate one token, recomputing the whole sequence",
    xaxis_title="sequence length (tokens)",
    yaxis_title="ms per token",
    yaxis_rangemode="tozero",
    height=380,
)
viz.show(fig)
```

```text
   16 tokens:   82.2 ms per new token
   64 tokens:  104.0 ms per new token
  256 tokens:  101.7 ms per new token
  512 tokens:  186.9 ms per new token
 1024 tokens:  570.1 ms per new token
 2048 tokens: 1906.7 ms per new token
```

<div class="plotly-figure" data-src="generated/01_07_sampling_20_6.json"></div>

In the run shown here, on an Apple-silicon Mac with a Metal GPU, a step took
about 80 to 105 ms up to 256 tokens, about 570 ms at 1,024 tokens, and about 1.9
s at 2,048. Treat these numbers as an example: yours depend on your hardware and
vary from run to run. For short sequences the step costs about the same at any
length, because fixed work, such as reading every weight and launching each
kernel, dominates. Past a few hundred tokens the cost climbs faster than the
length itself, because attention compares every position with every earlier one.
Other devices give other numbers with the same shape.

The keys and values of earlier positions never change, so all the work of
recomputing them is wasted.
[Section 1.8](01_08_kv_cache.md) caches them. The
handbook's
[LLM inference metrics](https://handbook.modular.com/llm-inference-basics/llm-inference-metrics/)
page explains why this per-token time, the TPOT from
[Section 1.1](01_01_run_and_serve.md), is the number
users feel while a reply streams in.

## Exercises

1. Sample the same prompt 5 times at `temperature=1.0` with and without
   `top_p=0.9`. How often does each run go off topic?
2. Implement *min-p* sampling: keep tokens whose probability is at least
   `min_p` times the top token's probability. Which MAX operations do you
   need?
3. Plot how many tokens top-p keeps at each step of a generation. When is
   the model confident?

## Next

[Section 1.8](01_08_kv_cache.md) adds a KV cache so
each step processes one token instead of the whole sequence.

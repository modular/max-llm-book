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

<details class="code-fold">
<summary>Show the setup <span class="code-fold-lines">32 lines</span></summary>

```python
import logging
import time
import warnings

import numpy as np
import plotly.graph_objects as go
from IPython.display import Markdown, display
from plotly.subplots import make_subplots
from max import functional as F
from max.driver import CPU
from max.dtype import DType
from max.functional import random
from max.graph import TensorType
from max.pipelines import AutoModel, AutoTokenizer
from max.tensor import Tensor

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

</details>

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
```

<details class="code-fold">
<summary>Show the code <span class="code-fold-lines">26 lines</span></summary>

```python
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

</details>

<div class="plotly-figure" data-src="generated/01_07_sampling_11_0.json"></div>

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
sampled: [0.546 0.209 0.138 0.079 0.028]
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

<details class="code-fold">
<summary>Show <code>sample</code>, <code>generate</code> <span class="code-fold-lines">57 lines</span></summary>

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

</details>

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
```

<details class="code-fold">
<summary>Show the code <span class="code-fold-lines">58 lines</span></summary>

```python
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

</details>

<div class="plotly-figure" data-src="generated/01_07_sampling_18_0.json"></div>

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

Why does a high temperature drift? Color each generated token by the
probability it had in the distribution it was drawn from (temperature 1 for
the greedy run), and hover over a chip for the five likeliest alternatives:

```python
def generate_with_probs(
    prompt: list[int], n: int, temperature: float
) -> tuple[list[int], list[np.ndarray]]:
    """Generates like `generate`, keeping each step's sampling distribution."""
    ids, dists = list(prompt), []
    for _ in range(n):
        logits = next_logits(ids)
        dists.append(F.softmax(logits / (temperature or 1.0)).to_numpy())
        token = sample(logits, temperature=temperature)
        ids.append(token)
        if token == config.eos_token_id:
            break
    return ids[len(prompt) :], dists


runs = {}
for temperature in [0, 1.5]:
    random.set_seed(0)
    runs[temperature] = generate_with_probs(story, 24, temperature)
```

<details class="code-fold">
<summary>Show <code>chip_title</code> <span class="code-fold-lines">20 lines</span></summary>

```python
def chip_title(token: int, dist: np.ndarray) -> str:
    likeliest = np.argsort(dist)[::-1][:5]
    alternatives = ", ".join(
        f"{tokenizer.decode([int(i)])!r} {dist[i]:.2f}" for i in likeliest
    )
    return f"p = {dist[token]:.2f}; likeliest: {alternatives}"


for temperature, (new, dists) in runs.items():
    chosen = [float(d[t]) for t, d in zip(new, dists)]
    mean = np.mean(chosen)
    display(Markdown(f"`temperature={temperature}`, mean p = {mean:.2f}:"))
    display(
        viz.tokens(
            [tokenizer.decode([t]) for t in new],
            new,
            colors=[f"rgba(42,120,214,{0.08 + 0.72 * p:.2f})" for p in chosen],
            titles=[chip_title(t, d) for t, d in zip(new, dists)],
        )
    )
```

</details>

`temperature=0`, mean p = 0.56:

<div style="line-height:2"><span title="p = 0.21; likeliest: &#x27; young&#x27; 0.21, &#x27; man&#x27; 0.12, &#x27; small&#x27; 0.06, &#x27; woman&#x27; 0.04, &#x27; great&#x27; 0.03" style="background:rgba(42,120,214,0.23);border-radius:4px;padding:2px 4px;margin:0 2px 4px 0;display:inline-block;font-family:ui-monospace,monospace;font-size:0.9em">&middot;young</span><span title="p = 0.23; likeliest: &#x27; woman&#x27; 0.23, &#x27; girl&#x27; 0.23, &#x27; man&#x27; 0.14, &#x27; boy&#x27; 0.08, &#x27; and&#x27; 0.02" style="background:rgba(42,120,214,0.25);border-radius:4px;padding:2px 4px;margin:0 2px 4px 0;display:inline-block;font-family:ui-monospace,monospace;font-size:0.9em">&middot;woman</span><span title="p = 0.94; likeliest: &#x27; named&#x27; 0.94, &#x27; who&#x27; 0.03, &#x27;,&#x27; 0.02, &#x27; with&#x27; 0.00, &#x27; in&#x27; 0.00" style="background:rgba(42,120,214,0.75);border-radius:4px;padding:2px 4px;margin:0 2px 4px 0;display:inline-block;font-family:ui-monospace,monospace;font-size:0.9em">&middot;named</span><span title="p = 0.10; likeliest: &#x27; Sarah&#x27; 0.10, &#x27; Sophia&#x27; 0.10, &#x27; Emma&#x27; 0.09, &#x27; A&#x27; 0.06, &#x27; Av&#x27; 0.05" style="background:rgba(42,120,214,0.16);border-radius:4px;padding:2px 4px;margin:0 2px 4px 0;display:inline-block;font-family:ui-monospace,monospace;font-size:0.9em">&middot;Sarah</span><span title="p = 0.52; likeliest: &#x27; who&#x27; 0.52, &#x27;.&#x27; 0.26, &#x27;,&#x27; 0.20, &#x27; living&#x27; 0.00, &#x27; from&#x27; 0.00" style="background:rgba(42,120,214,0.45);border-radius:4px;padding:2px 4px;margin:0 2px 4px 0;display:inline-block;font-family:ui-monospace,monospace;font-size:0.9em">&middot;who</span><span title="p = 0.27; likeliest: &#x27; was&#x27; 0.27, &#x27; had&#x27; 0.23, &#x27; lived&#x27; 0.18, &#x27; worked&#x27; 0.04, &#x27; loved&#x27; 0.04" style="background:rgba(42,120,214,0.27);border-radius:4px;padding:2px 4px;margin:0 2px 4px 0;display:inline-block;font-family:ui-monospace,monospace;font-size:0.9em">&middot;was</span><span title="p = 0.19; likeliest: &#x27; struggling&#x27; 0.19, &#x27; a&#x27; 0.15, &#x27; working&#x27; 0.05, &#x27; an&#x27; 0.04, &#x27; on&#x27; 0.02" style="background:rgba(42,120,214,0.21);border-radius:4px;padding:2px 4px;margin:0 2px 4px 0;display:inline-block;font-family:ui-monospace,monospace;font-size:0.9em">&middot;struggling</span><span title="p = 0.72; likeliest: &#x27; to&#x27; 0.72, &#x27; with&#x27; 0.25, &#x27; in&#x27; 0.02, &#x27; financially&#x27; 0.01, &#x27;.&#x27; 0.01" style="background:rgba(42,120,214,0.60);border-radius:4px;padding:2px 4px;margin:0 2px 4px 0;display:inline-block;font-family:ui-monospace,monospace;font-size:0.9em">&middot;to</span><span title="p = 0.52; likeliest: &#x27; find&#x27; 0.52, &#x27; make&#x27; 0.21, &#x27; balance&#x27; 0.05, &#x27; decide&#x27; 0.03, &#x27; get&#x27; 0.02" style="background:rgba(42,120,214,0.45);border-radius:4px;padding:2px 4px;margin:0 2px 4px 0;display:inline-block;font-family:ui-monospace,monospace;font-size:0.9em">&middot;find</span><span title="p = 0.81; likeliest: &#x27; her&#x27; 0.81, &#x27; a&#x27; 0.09, &#x27; happiness&#x27; 0.01, &#x27; meaning&#x27; 0.01, &#x27; work&#x27; 0.01" style="background:rgba(42,120,214,0.66);border-radius:4px;padding:2px 4px;margin:0 2px 4px 0;display:inline-block;font-family:ui-monospace,monospace;font-size:0.9em">&middot;her</span><span title="p = 0.84; likeliest: &#x27; place&#x27; 0.84, &#x27; voice&#x27; 0.08, &#x27; way&#x27; 0.01, &#x27; passion&#x27; 0.01, &#x27; own&#x27; 0.01" style="background:rgba(42,120,214,0.69);border-radius:4px;padding:2px 4px;margin:0 2px 4px 0;display:inline-block;font-family:ui-monospace,monospace;font-size:0.9em">&middot;place</span><span title="p = 0.90; likeliest: &#x27; in&#x27; 0.90, &#x27;.&#x27; 0.06, &#x27; within&#x27; 0.01, &#x27; and&#x27; 0.01, &#x27; at&#x27; 0.01" style="background:rgba(42,120,214,0.73);border-radius:4px;padding:2px 4px;margin:0 2px 4px 0;display:inline-block;font-family:ui-monospace,monospace;font-size:0.9em">&middot;in</span><span title="p = 0.61; likeliest: &#x27; the&#x27; 0.61, &#x27; a&#x27; 0.19, &#x27; her&#x27; 0.09, &#x27; society&#x27; 0.04, &#x27; this&#x27; 0.02" style="background:rgba(42,120,214,0.52);border-radius:4px;padding:2px 4px;margin:0 2px 4px 0;display:inline-block;font-family:ui-monospace,monospace;font-size:0.9em">&middot;the</span><span title="p = 0.90; likeliest: &#x27; world&#x27; 0.90, &#x27; community&#x27; 0.01, &#x27; city&#x27; 0.01, &#x27; bustling&#x27; 0.00, &#x27; corporate&#x27; 0.00" style="background:rgba(42,120,214,0.73);border-radius:4px;padding:2px 4px;margin:0 2px 4px 0;display:inline-block;font-family:ui-monospace,monospace;font-size:0.9em">&middot;world</span><span title="p = 0.96; likeliest: &#x27;.&#x27; 0.96, &#x27; of&#x27; 0.01, &#x27; and&#x27; 0.01, &#x27; around&#x27; 0.01, &#x27;,&#x27; 0.01" style="background:rgba(42,120,214,0.77);border-radius:4px;padding:2px 4px;margin:0 2px 4px 0;display:inline-block;font-family:ui-monospace,monospace;font-size:0.9em">.</span><span title="p = 0.94; likeliest: &#x27; She&#x27; 0.94, &#x27; Her&#x27; 0.03, &#x27; Sarah&#x27; 0.01, &#x27; Despite&#x27; 0.00, &#x27; The&#x27; 0.00" style="background:rgba(42,120,214,0.76);border-radius:4px;padding:2px 4px;margin:0 2px 4px 0;display:inline-block;font-family:ui-monospace,monospace;font-size:0.9em">&middot;She</span><span title="p = 0.44; likeliest: &#x27; had&#x27; 0.44, &#x27; was&#x27; 0.29, &#x27; felt&#x27; 0.11, &#x27; loved&#x27; 0.05, &#x27; lived&#x27; 0.03" style="background:rgba(42,120,214,0.40);border-radius:4px;padding:2px 4px;margin:0 2px 4px 0;display:inline-block;font-family:ui-monospace,monospace;font-size:0.9em">&middot;had</span><span title="p = 0.77; likeliest: &#x27; always&#x27; 0.77, &#x27; been&#x27; 0.09, &#x27; a&#x27; 0.04, &#x27; just&#x27; 0.01, &#x27; no&#x27; 0.01" style="background:rgba(42,120,214,0.63);border-radius:4px;padding:2px 4px;margin:0 2px 4px 0;display:inline-block;font-family:ui-monospace,monospace;font-size:0.9em">&middot;always</span><span title="p = 0.34; likeliest: &#x27; been&#x27; 0.34, &#x27; dreamed&#x27; 0.32, &#x27; felt&#x27; 0.19, &#x27; wanted&#x27; 0.05, &#x27; loved&#x27; 0.03" style="background:rgba(42,120,214,0.32);border-radius:4px;padding:2px 4px;margin:0 2px 4px 0;display:inline-block;font-family:ui-monospace,monospace;font-size:0.9em">&middot;been</span><span title="p = 0.15; likeliest: &#x27; a&#x27; 0.15, &#x27; drawn&#x27; 0.15, &#x27; passionate&#x27; 0.12, &#x27; the&#x27; 0.05, &#x27; fascinated&#x27; 0.05" style="background:rgba(42,120,214,0.19);border-radius:4px;padding:2px 4px;margin:0 2px 4px 0;display:inline-block;font-family:ui-monospace,monospace;font-size:0.9em">&middot;a</span><span title="p = 0.19; likeliest: &#x27; bit&#x27; 0.19, &#x27; quiet&#x27; 0.11, &#x27; creative&#x27; 0.08, &#x27; shy&#x27; 0.07, &#x27; free&#x27; 0.04" style="background:rgba(42,120,214,0.22);border-radius:4px;padding:2px 4px;margin:0 2px 4px 0;display:inline-block;font-family:ui-monospace,monospace;font-size:0.9em">&middot;bit</span><span title="p = 0.93; likeliest: &#x27; of&#x27; 0.93, &#x27; reserved&#x27; 0.01, &#x27; shy&#x27; 0.01, &#x27; awkward&#x27; 0.01, &#x27; unconventional&#x27; 0.01" style="background:rgba(42,120,214,0.75);border-radius:4px;padding:2px 4px;margin:0 2px 4px 0;display:inline-block;font-family:ui-monospace,monospace;font-size:0.9em">&middot;of</span><span title="p = 0.87; likeliest: &#x27; a&#x27; 0.87, &#x27; an&#x27; 0.13, &#x27; trouble&#x27; 0.00, &#x27; someone&#x27; 0.00, &#x27; of&#x27; 0.00" style="background:rgba(42,120,214,0.71);border-radius:4px;padding:2px 4px;margin:0 2px 4px 0;display:inline-block;font-family:ui-monospace,monospace;font-size:0.9em">&middot;a</span><span title="p = 0.15; likeliest: &#x27; rebel&#x27; 0.15, &#x27; free&#x27; 0.14, &#x27; lon&#x27; 0.13, &#x27; intro&#x27; 0.06, &#x27; shy&#x27; 0.05" style="background:rgba(42,120,214,0.19);border-radius:4px;padding:2px 4px;margin:0 2px 4px 0;display:inline-block;font-family:ui-monospace,monospace;font-size:0.9em">&middot;rebel</span></div>

`temperature=1.5`, mean p = 0.01:

<div style="line-height:2"><span title="p = 0.00; likeliest: &#x27; young&#x27; 0.05, &#x27; man&#x27; 0.03, &#x27; small&#x27; 0.02, &#x27; woman&#x27; 0.02, &#x27; great&#x27; 0.01" style="background:rgba(42,120,214,0.08);border-radius:4px;padding:2px 4px;margin:0 2px 4px 0;display:inline-block;font-family:ui-monospace,monospace;font-size:0.9em">&middot;neighboring</span><span title="p = 0.00; likeliest: &#x27; kingdom&#x27; 0.12, &#x27; town&#x27; 0.10, &#x27; city&#x27; 0.07, &#x27; country&#x27; 0.05, &#x27; nation&#x27; 0.04" style="background:rgba(42,120,214,0.08);border-radius:4px;padding:2px 4px;margin:0 2px 4px 0;display:inline-block;font-family:ui-monospace,monospace;font-size:0.9em">&middot;realm</span><span title="p = 0.01; likeliest: &#x27; called&#x27; 0.20, &#x27;,&#x27; 0.18, &#x27; that&#x27; 0.08, &#x27; known&#x27; 0.08, &#x27;.&#x27; 0.04" style="background:rgba(42,120,214,0.08);border-radius:4px;padding:2px 4px;margin:0 2px 4px 0;display:inline-block;font-family:ui-monospace,monospace;font-size:0.9em">&middot;on</span><span title="p = 0.00; likeliest: &#x27; the&#x27; 0.47, &#x27; a&#x27; 0.09, &#x27; Earth&#x27; 0.04, &#x27; which&#x27; 0.02, &#x27; an&#x27; 0.02" style="background:rgba(42,120,214,0.08);border-radius:4px;padding:2px 4px;margin:0 2px 4px 0;display:inline-block;font-family:ui-monospace,monospace;font-size:0.9em">&middot;your</span><span title="p = 0.02; likeliest: &#x27; side&#x27; 0.09, &#x27; own&#x27; 0.03, &#x27; journey&#x27; 0.02, &#x27; home&#x27; 0.02, &#x27; planet&#x27; 0.02" style="background:rgba(42,120,214,0.09);border-radius:4px;padding:2px 4px;margin:0 2px 4px 0;display:inline-block;font-family:ui-monospace,monospace;font-size:0.9em">&middot;planet</span><span title="p = 0.19; likeliest: &#x27;,&#x27; 0.19, &#x27;.&#x27; 0.19, &#x27; that&#x27; 0.07, &quot;&#x27;s&quot; 0.04, &#x27; called&#x27; 0.04" style="background:rgba(42,120,214,0.22);border-radius:4px;padding:2px 4px;margin:0 2px 4px 0;display:inline-block;font-family:ui-monospace,monospace;font-size:0.9em">,</span><span title="p = 0.00; likeliest: &#x27; where&#x27; 0.06, &#x27; and&#x27; 0.05, &#x27; a&#x27; 0.05, &#x27; known&#x27; 0.04, &#x27; one&#x27; 0.04" style="background:rgba(42,120,214,0.08);border-radius:4px;padding:2px 4px;margin:0 2px 4px 0;display:inline-block;font-family:ui-monospace,monospace;font-size:0.9em">&middot;mist</span><span title="p = 0.01; likeliest: &#x27;y&#x27; 0.17, &#x27;-&#x27; 0.10, &#x27;ing&#x27; 0.02, &#x27; and&#x27; 0.02, &#x27;bound&#x27; 0.02" style="background:rgba(42,120,214,0.09);border-radius:4px;padding:2px 4px;margin:0 2px 4px 0;display:inline-block;font-family:ui-monospace,monospace;font-size:0.9em">.</span><span title="p = 0.00; likeliest: &#x27; The&#x27; 0.05, &#x27; It&#x27; 0.04, &#x27; This&#x27; 0.03, &#x27; You&#x27; 0.03, &#x27; In&#x27; 0.02" style="background:rgba(42,120,214,0.08);border-radius:4px;padding:2px 4px;margin:0 2px 4px 0;display:inline-block;font-family:ui-monospace,monospace;font-size:0.9em">&middot;Even</span><span title="p = 0.01; likeliest: &#x27; though&#x27; 0.11, &#x27; now&#x27; 0.04, &#x27; the&#x27; 0.04, &#x27; in&#x27; 0.03, &#x27; as&#x27; 0.02" style="background:rgba(42,120,214,0.09);border-radius:4px;padding:2px 4px;margin:0 2px 4px 0;display:inline-block;font-family:ui-monospace,monospace;font-size:0.9em">&middot;then</span><span title="p = 0.00; likeliest: &#x27;,&#x27; 0.42, &#x27; you&#x27; 0.03, &#x27; there&#x27; 0.02, &#x27; the&#x27; 0.02, &#x27; I&#x27; 0.02" style="background:rgba(42,120,214,0.08);border-radius:4px;padding:2px 4px;margin:0 2px 4px 0;display:inline-block;font-family:ui-monospace,monospace;font-size:0.9em">;</span><span title="p = 0.00; likeliest: &#x27; you&#x27; 0.03, &#x27; the&#x27; 0.02, &#x27; it&#x27; 0.02, &#x27; there&#x27; 0.02, &#x27; I&#x27; 0.01" style="background:rgba(42,120,214,0.08);border-radius:4px;padding:2px 4px;margin:0 2px 4px 0;display:inline-block;font-family:ui-monospace,monospace;font-size:0.9em">&middot;Unlocking</span><span title="p = 0.00; likeliest: &#x27; the&#x27; 0.08, &#x27; was&#x27; 0.02, &#x27; your&#x27; 0.02, &#x27;.&#x27; 0.01, &#x27; The&#x27; 0.01" style="background:rgba(42,120,214,0.08);border-radius:4px;padding:2px 4px;margin:0 2px 4px 0;display:inline-block;font-family:ui-monospace,monospace;font-size:0.9em">&middot;beast</span><span title="p = 0.00; likeliest: &#x27;.&#x27; 0.03, &#x27;,&#x27; 0.02, &#x27; of&#x27; 0.01, &#x27; was&#x27; 0.01, &quot;&#x27;s&quot; 0.01" style="background:rgba(42,120,214,0.08);border-radius:4px;padding:2px 4px;margin:0 2px 4px 0;display:inline-block;font-family:ui-monospace,monospace;font-size:0.9em">ell</span><span title="p = 0.00; likeliest: &quot;&#x27;s&quot; 0.01, &#x27;.&#x27; 0.01, &#x27;,&#x27; 0.01, &#x27; was&#x27; 0.01, &#x27; were&#x27; 0.01" style="background:rgba(42,120,214,0.08);border-radius:4px;padding:2px 4px;margin:0 2px 4px 0;display:inline-block;font-family:ui-monospace,monospace;font-size:0.9em">ands</span><span title="p = 0.00; likeliest: &#x27;,&#x27; 0.03, &#x27; was&#x27; 0.02, &#x27;.&#x27; 0.02, &#x27; is&#x27; 0.01, &#x27; were&#x27; 0.01" style="background:rgba(42,120,214,0.08);border-radius:4px;padding:2px 4px;margin:0 2px 4px 0;display:inline-block;font-family:ui-monospace,monospace;font-size:0.9em">is</span><span title="p = 0.00; likeliest: &#x27;.&#x27; 0.02, &#x27; was&#x27; 0.01, &#x27;,&#x27; 0.01, &#x27; is&#x27; 0.01, &#x27;\n&#x27; 0.01" style="background:rgba(42,120,214,0.08);border-radius:4px;padding:2px 4px;margin:0 2px 4px 0;display:inline-block;font-family:ui-monospace,monospace;font-size:0.9em">&middot;death</span><span title="p = 0.00; likeliest: &#x27;.&#x27; 0.03, &#x27;,&#x27; 0.02, &#x27; and&#x27; 0.01, &#x27; to&#x27; 0.01, &#x27; was&#x27; 0.01" style="background:rgba(42,120,214,0.08);border-radius:4px;padding:2px 4px;margin:0 2px 4px 0;display:inline-block;font-family:ui-monospace,monospace;font-size:0.9em">shire</span><span title="p = 0.00; likeliest: &#x27;.&#x27; 0.02, &#x27;,&#x27; 0.01, &#x27;\n&#x27; 0.01, &#x27; was&#x27; 0.01, &#x27; and&#x27; 0.01" style="background:rgba(42,120,214,0.08);border-radius:4px;padding:2px 4px;margin:0 2px 4px 0;display:inline-block;font-family:ui-monospace,monospace;font-size:0.9em">&middot;i</span><span title="p = 0.00; likeliest: &#x27;.&#x27; 0.01, &#x27; am&#x27; 0.01, &#x27; was&#x27; 0.01, &#x27; would&#x27; 0.01, &#x27; have&#x27; 0.01" style="background:rgba(42,120,214,0.08);border-radius:4px;padding:2px 4px;margin:0 2px 4px 0;display:inline-block;font-family:ui-monospace,monospace;font-size:0.9em">sts</span><span title="p = 0.00; likeliest: &#x27;.&#x27; 0.01, &#x27; the&#x27; 0.00, &#x27;,&#x27; 0.00, &#x27; a&#x27; 0.00, &#x27; and&#x27; 0.00" style="background:rgba(42,120,214,0.08);border-radius:4px;padding:2px 4px;margin:0 2px 4px 0;display:inline-block;font-family:ui-monospace,monospace;font-size:0.9em">ub</span><span title="p = 0.00; likeliest: &#x27;st&#x27; 0.02, &#x27;ject&#x27; 0.02, &#x27;sc&#x27; 0.02, &#x27;scription&#x27; 0.02, &#x27;scribed&#x27; 0.01" style="background:rgba(42,120,214,0.08);border-radius:4px;padding:2px 4px;margin:0 2px 4px 0;display:inline-block;font-family:ui-monospace,monospace;font-size:0.9em">bral</span><span title="p = 0.00; likeliest: &#x27;.&#x27; 0.01, &#x27;,&#x27; 0.00, &#x27; and&#x27; 0.00, &#x27;\n&#x27; 0.00, &#x27;ly&#x27; 0.00" style="background:rgba(42,120,214,0.08);border-radius:4px;padding:2px 4px;margin:0 2px 4px 0;display:inline-block;font-family:ui-monospace,monospace;font-size:0.9em">gen</span><span title="p = 0.00; likeliest: &#x27;i&#x27; 0.01, &#x27;.&#x27; 0.01, &#x27;ii&#x27; 0.01, &#x27;ial&#x27; 0.01, &#x27;ies&#x27; 0.01" style="background:rgba(42,120,214,0.08);border-radius:4px;padding:2px 4px;margin:0 2px 4px 0;display:inline-block;font-family:ui-monospace,monospace;font-size:0.9em">aki</span></div>

Higher temperatures read less predictably. The greedy run picks tokens with
probabilities from about 0.1 to over 0.9. At 1.5 without a filter, nearly
every chip is pale: a flattened distribution gives the long tail of unlikely
tokens enough combined weight that most picks are tokens the model gave
almost no chance, and the text turns to gibberish within a few words. Many chat applications use settings near
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
    for _ in range(5):
        start = time.perf_counter()
        next_logits(ids).to_numpy()
        times.append((time.perf_counter() - start) * 1000)
    step_ms.append(float(np.median(times)))
    print(f"{length:5} tokens: {step_ms[-1]:6.1f} ms per new token")
```

```text
   16 tokens:   57.7 ms per new token
   64 tokens:   60.1 ms per new token
  256 tokens:   70.9 ms per new token
  512 tokens:  179.8 ms per new token
 1024 tokens:  563.3 ms per new token
 2048 tokens: 1839.3 ms per new token
```

<details class="code-fold">
<summary>Show the code <span class="code-fold-lines">17 lines</span></summary>

```python
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

</details>

<div class="plotly-figure" data-src="generated/01_07_sampling_26_0.json"></div>

In the run shown here, on an Apple-silicon Mac with a Metal GPU, a step took
about 70 to 110 ms up to 256 tokens, about 560 ms at 1,024 tokens, and about 1.8
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

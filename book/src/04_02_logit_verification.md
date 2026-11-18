# 4.2 Verifying a model's logits

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/modular/max-llm-book/blob/main/notebooks/04_02_logit_verification.ipynb)
[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/modular/max-llm-book/blob/main/notebooks/04_02_logit_verification.ipynb)
[![View on GitHub](https://img.shields.io/badge/GitHub-notebook-black?logo=github)](https://github.com/modular/max-llm-book/blob/main/notebooks/04_02_logit_verification.ipynb)

Layer tests catch bugs in one layer. A whole model can still go wrong in
ways no layer test sees: a weight loaded into the wrong layer, a tied head
that isn't tied, a mask that only fails on long prompts. So model bring-ups
end with *logit verification*: run the same prompts through your model and a
trusted reference, and compare the logits at every position.

This section verifies your SmolLM2 against Hugging Face's PyTorch
implementation, with the metrics MAX's own verification tools use.

<details class="code-fold">
<summary>Show the setup <span class="code-fold-lines">36 lines</span></summary>

```python
import logging
import warnings

import numpy as np
import plotly.graph_objects as go
import torch
from max.driver import CPU
from max.dtype import DType
from max.graph import TensorType
from max.pipelines import AutoTokenizer
from max.tensor import Tensor
from transformers import AutoModelForCausalLM

from minillm import viz
from minillm.device import pick_device
from minillm.config import MODEL_ID, LlamaConfig
from minillm.kv_cache import CachedLlamaForCausalLM, KVCacheDecoder
from minillm.model import load_model
from minillm.sampling import generate
from minillm.testing import compare_outputs
from minillm.weights import load_checkpoint

logging.getLogger("max").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="The eager interpreter failed")
DEVICE = pick_device()  # a GPU if MAX finds one, otherwise the CPU

config = LlamaConfig.from_pretrained()
weights = load_checkpoint()
tokenizer = AutoTokenizer(MODEL_ID)

ours = load_model(config, weights, DEVICE).compile(
    TensorType(DType.int64, ("batch", "seq_len"), device=DEVICE)
)
reference = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.float32
).eval()
```

</details>

## Prompts

Verification prompts should exercise different behavior: plain prose, code,
numbers, another language, and a prompt long enough to stress attention and
RoPE at larger positions.

```python
long_prompt = " ".join(["The quick brown fox jumps over the lazy dog."] * 60)
prompts = {
    "prose": "The capital of France is Paris, and the capital of Italy is",
    "code": "def fibonacci(n):\n    if n < 2:\n        return n\n    return",
    "numbers": "1, 1, 2, 3, 5, 8, 13, 21,",
    "french": "Le chat est assis sur",
    "long": long_prompt,
}


def logits_pair(text: str) -> tuple[np.ndarray, np.ndarray]:
    ids = tokenizer([text])["input_ids"][0]
    mine = ours(Tensor(np.array([ids]), device=DEVICE)).to_numpy()[0]
    with torch.no_grad():
        theirs = reference(torch.tensor([ids])).logits[0].numpy()
    return mine, theirs
```

## Metrics for logits

Logits feed a softmax, so on top of the elementwise metrics from
[Section 4.1](04_01_layer_tests.md), two measures
matter for generation.

The *KL divergence* between the next-token distributions measures how much
probability mass moved. It is the standard measure for comparing two models'
predictions, and it ignores differences in logits that softmax removes, such
as a constant offset. For the reference distribution \\(P\\) and yours, \\(Q\\), at
one position:

\\[D\_{\\mathrm{KL}}(P \\,\\Vert\\, Q) = \\sum\_i P\_i \\log \\frac{P\_i}{Q\_i}\\]

It is 0 only when the two distributions are identical.

*Top-1 agreement* is the fraction of positions at which both models predict
the same most likely token. It is all that greedy decoding sees.

<details class="code-fold">
<summary>Show <code>softmax</code>, <code>kl_divergence</code> <span class="code-fold-lines">26 lines</span></summary>

```python
def softmax(z: np.ndarray) -> np.ndarray:
    z = z.astype(np.float64) - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def kl_divergence(p_logits: np.ndarray, q_logits: np.ndarray) -> float:
    """Mean KL(P || Q) over positions, in nats."""
    p, q = softmax(p_logits), softmax(q_logits)
    return float((p * (np.log(p + 1e-30) - np.log(q + 1e-30))).sum(-1).mean())


print(
    f"{'prompt':8} {'positions':>9} {'max err':>8} {'cos dist':>9} {'KL':>9} {'top-1':>6}"
)
errors = {}
for name, text in prompts.items():
    mine, theirs = logits_pair(text)
    errors[name] = np.abs(mine - theirs).max(-1)
    r = compare_outputs(mine, theirs, atol=1e-3, rtol=1e-3)
    top1 = (mine.argmax(-1) == theirs.argmax(-1)).mean()
    print(
        f"{name:8} {len(mine):>9} {r.max_abs_err:>8.1e} {r.cos_dist:>9.1e} "
        f"{kl_divergence(theirs, mine):>9.1e} {top1:>6.0%}"
    )
    assert top1 >= 0.99 and kl_divergence(theirs, mine) < 1e-6
```

</details>

```text
prompt   positions  max err  cos dist        KL  top-1
prose           13  8.2e-05   2.4e-12   3.2e-11   100%
code            18  8.3e-05   1.5e-12   4.0e-11   100%
numbers         25  1.1e-04   2.1e-12   2.2e-11   100%
french           6  1.0e-04   1.3e-12   8.6e-11   100%
long           600  1.1e-04   2.6e-12   2.5e-12   100%
```

A summary number can hide a trend. Plot the largest logit error at each
position: if a bug in RoPE or the mask grew with position, the long prompt
would show it as a rising line.

<details class="code-fold">
<summary>Show the code <span class="code-fold-lines">22 lines</span></summary>

```python
fig = go.Figure(
    [
        go.Scatter(
            x=np.arange(1, len(err) + 1),
            y=err,
            customdata=np.arange(len(err)),
            name=name,
            mode="lines",
            line_color=color,
            hovertemplate=name
            + ", position %{customdata}: %{y:.1e}<extra></extra>",
        )
        for (name, err), color in zip(errors.items(), viz.SERIES)
    ]
)
fig.update_layout(
    title="Largest logit error at each position, against Hugging Face",
    height=400,
    xaxis=dict(title="position + 1 (log scale)", type="log", dtick=1),
    yaxis=dict(title="max |logit difference|", type="log", dtick=1),
)
viz.show(fig)
```

</details>

<div class="plotly-figure" data-src="generated/04_02_logit_verification_8_0.json"></div>

Every prompt agrees at every position, including the 600-token one, with 100%
top-1 agreement and KL divergences between about `1e-12` and `1e-10` nats. MAX
verifies its own models the same way, comparing MAX's logits with reference
logits from PyTorch or vLLM by cosine distance, KL divergence, and tolerances.
The
[model bring-up workflow](https://max.modular.com/develop/model-bringup-workflow)
guide covers the rest of bringing a new architecture to MAX.

## Greedy generation

Matching logits at the prompt is necessary but not sufficient: generation feeds
the model its own outputs, through your KV cache from
[Section 1.8](01_08_kv_cache.md). Compare 48 greedy
tokens from your cached decoder with Hugging Face's `generate`:

<details class="code-fold">
<summary>Show the code <span class="code-fold-lines">27 lines</span></summary>

```python
cached = KVCacheDecoder(
    load_model(config, weights, DEVICE, CachedLlamaForCausalLM), config, DEVICE
)

gaps = {}
for name in ["prose", "code", "numbers"]:
    ids = tokenizer([prompts[name]])["input_ids"][0]
    mine = generate(cached, ids, 48, temperature=0)
    with torch.no_grad():
        out = reference.generate(
            torch.tensor([ids]),
            max_new_tokens=48,
            do_sample=False,
            output_logits=True,
            return_dict_in_generate=True,
        )
    theirs = out.sequences[0, len(ids) :].tolist()
    gaps[name] = [
        float(np.diff(np.sort(step[0].numpy())[-2:])[0]) for step in out.logits
    ]
    same = next(
        (i for i, (a, b) in enumerate(zip(mine, theirs)) if a != b), len(mine)
    )
    print(f"{name:8} identical for {same} of {len(mine)} tokens")
    assert same >= 24, (
        "diverged early; check the gap between the top two logits"
    )
```

</details>

```text
prose    identical for 48 of 48 tokens
code     identical for 48 of 48 tokens
numbers  identical for 48 of 48 tokens
```

At each greedy step, how far ahead was the winning token? `output_logits=True`
returns the reference's raw logits at every step; plot the gap between the top
two, with the largest logit error from the first chart as a dotted line:

<details class="code-fold">
<summary>Show the code <span class="code-fold-lines">32 lines</span></summary>

```python
fig = go.Figure(
    [
        go.Scatter(
            x=np.arange(len(gap)),
            y=gap,
            name=name,
            mode="lines+markers",
            line_color=color,
            marker=dict(size=5),
            hovertemplate=name + ", step %{x}: gap %{y:.3f}<extra></extra>",
        )
        for (name, gap), color in zip(gaps.items(), viz.SERIES)
    ]
)
fig.update_layout(
    title="Gap between the top two logits at each greedy step",
    height=400,
    xaxis_title="generated token",
    yaxis=dict(title="logit gap (log scale)", type="log", dtick=1),
)
largest_error = max(err.max() for err in errors.values())
fig.add_trace(
    go.Scatter(
        x=[0, max(len(gap) for gap in gaps.values()) - 1],
        y=[largest_error] * 2,
        name=f"largest logit error ({largest_error:.0e})",
        mode="lines",
        line=dict(color=viz.INK_SECONDARY, dash="dot", width=1),
        hoverinfo="skip",
    )
)
viz.show(fig)
```

</details>

<div class="plotly-figure" data-src="generated/04_02_logit_verification_12_0.json"></div>

A rounding difference can change the winner only at a step where the gap is
smaller than the error, that is, where a line dips below the dotted one. Here the smallest gap is above `0.1` and the largest error
about `1e-4`, so no step comes close.

If a run does diverge after many tokens, the cause is usually a *near tie*:
at some step two tokens have almost the same logit, and a difference in the
fourth or fifth decimal place, from summing in a different order, picks a
different winner. From then on the two texts differ, though both are valid
greedy continuations. When you see a divergence, look at the gap between the
top two logits at that step: a tiny gap means rounding, a large one means a
bug.

## Beyond logits: accuracy evaluations

Logit verification compares two implementations. It can't tell you whether the
model is *good* at a task, or whether a change that alters numbers on purpose,
such as quantization, costs accuracy. For that, serving teams run task
evaluations such as GSM8K or MMLU against the server
([Section 7.1](07_01_serve_your_model.md) serves your
model). The handbook's
[quantization](https://handbook.modular.com/model-preparation/llm-quantization/)
page covers how quantized models are evaluated.

## Exercises

1. Run the comparison in `bfloat16` (cast the weights and cast the logits
   back to `float32`). How do KL divergence and top-1 agreement change?
2. Find the step where two tokens' logits are closest during a greedy run.
   How small is the gap?
3. Load one layer's weights into the wrong layer index and rerun the table.
   Which prompt and metric show it most clearly?

## Next

[Section 4.3](04_03_kernel_benchmarks.md) measures
speed: how to benchmark kernels and layers so the numbers mean something.

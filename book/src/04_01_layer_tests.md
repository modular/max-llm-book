# 4.1 Testing layers against references

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/modular/max-llm-book/blob/main/notebooks/04_01_layer_tests.ipynb)
[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/modular/max-llm-book/blob/main/notebooks/04_01_layer_tests.ipynb)
[![View on GitHub](https://img.shields.io/badge/GitHub-notebook-black?logo=github)](https://github.com/modular/max-llm-book/blob/main/notebooks/04_01_layer_tests.ipynb)

[Chapter 1](ch01.md) checked each layer once, on one
input. Real tests check many shapes against an independent reference, with
metrics that tell a rounding difference from a bug. This section builds the
pattern MAX uses for its own layers, in
[`max/python/layer_benchmarks`](https://github.com/modular/modular/tree/main/max/python/layer_benchmarks),
and applies it to your decoder layer, with Hugging Face's PyTorch Llama as the
reference.

<details class="code-fold">
<summary>Show the setup <span class="code-fold-lines">29 lines</span></summary>

```python
import logging
import warnings

import numpy as np
import plotly.graph_objects as go
import torch
from max import functional as F
from max.driver import CPU
from max.dtype import DType
from max.tensor import Tensor, default_device, default_dtype
from max.tensor.tensor import TensorType
from transformers import AutoConfig
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaRotaryEmbedding,
)

from minillm import viz
from minillm.device import pick_device
from minillm.config import MODEL_ID, LlamaConfig
from minillm.model import DecoderLayer
from minillm.weights import load_checkpoint, select

logging.getLogger("max").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="The eager interpreter failed")
DEVICE = pick_device()  # a GPU if MAX finds one, otherwise the CPU

config = LlamaConfig.from_pretrained()
layer0 = select(load_checkpoint(), "model.layers.0.")
```

</details>

## How to compare outputs

One number is not enough. MAX's test harness reports three:

- The maximum absolute error catches any single element that is far off.
- The mean absolute error is the typical error, which tells a few outliers
  from a systematic drift.
- The cosine distance, \\(1 - \\cos\\theta\\) between the flattened outputs, is
  near 0 when the outputs point the same way even if their scale differs
  slightly, and large for a structural bug such as a wrong head layout.

A test passes when the outputs are `allclose`, within an absolute and a
relative tolerance, and the cosine distance is below a threshold. The defaults
below, `1e-3` for both tolerances and `1e-5` for the cosine distance, are
tighter than MAX's (`1e-2` and `1e-3`) because every test here runs in
`float32`; MAX's `bfloat16` layer tests loosen them further.

<details class="code-fold">
<summary>Show <code>CorrectnessResult</code>, <code>compare_outputs</code> <span class="code-fold-lines">48 lines</span></summary>

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CorrectnessResult:
    """How far an output is from its reference, as MAX's layer tests report it."""

    label: str
    mae: float
    max_abs_err: float
    cos_dist: float
    passed: bool

    def __str__(self) -> str:
        status = "pass" if self.passed else "FAIL"
        return (
            f"{self.label:16} mae {self.mae:.1e}  max {self.max_abs_err:.1e}  "
            f"cos dist {self.cos_dist:.1e}  {status}"
        )


def compare_outputs(
    actual: np.ndarray,
    expected: np.ndarray,
    label: str = "",
    atol: float = 1e-3,
    rtol: float = 1e-3,
    cos_threshold: float = 1e-5,
) -> CorrectnessResult:
    """Compares ``actual`` with ``expected``; passes when they are allclose
    and their cosine distance is below ``cos_threshold``."""
    a = actual.astype(np.float64).ravel()
    e = expected.astype(np.float64).ravel()
    cos_dist = 1.0 - float(
        a @ e / (np.linalg.norm(a) * np.linalg.norm(e) + 1e-12)
    )
    allclose = bool(np.allclose(actual, expected, atol=atol, rtol=rtol))
    return CorrectnessResult(
        label=label,
        mae=float(np.mean(np.abs(a - e))),
        max_abs_err=float(np.max(np.abs(a - e))),
        cos_dist=cos_dist,
        passed=allclose and cos_dist < cos_threshold,
    )
```

</details>

<small>Saved to `minillm/testing.py`.</small>

## The reference

Hugging Face's `LlamaDecoderLayer` is an independent implementation of the
same layer, in PyTorch. Its parameter names match yours, so the same
checkpoint entries load into both. `"eager"` selects its plain attention
code instead of a fused kernel, so the reference stays simple.

```python
hf_config = AutoConfig.from_pretrained(MODEL_ID)
hf_config._attn_implementation = "eager"
reference = LlamaDecoderLayer(hf_config, layer_idx=0).float().eval()
reference.load_state_dict(
    {name: torch.from_numpy(w.copy()) for name, w in layer0.items()}
)
rotary = LlamaRotaryEmbedding(hf_config)


def reference_layer(x: np.ndarray) -> np.ndarray:
    batch, seq, _ = x.shape
    hidden = torch.from_numpy(x)
    positions = torch.arange(seq).expand(batch, seq)
    mask = torch.full((seq, seq), float("-inf")).triu(1)[None, None]
    with torch.no_grad():
        out = reference(
            hidden,
            attention_mask=mask,
            position_ids=positions,
            position_embeddings=rotary(hidden, positions),
        )
    return out.numpy()
```

## Compile once, test many shapes

MAX's harness separates *static* parameters, which fix the graph (here, the
layer's sizes), from *dynamic* ones, which vary per run (batch size and
sequence length). Compile once with symbolic dimensions, then run every
shape through the same compiled layer. The shapes cover the edges: a single
token (a decode step), odd lengths, a batch, and a long prefill.

<details class="code-fold">
<summary>Show <code>compiled_layer</code>, <code>check</code> <span class="code-fold-lines">34 lines</span></summary>

```python
def compiled_layer(layer_class=DecoderLayer, weights=layer0, cfg=config):
    with F.lazy(), default_device(DEVICE), default_dtype(DType.float32):
        layer = layer_class(cfg)
    layer.load_state_dict(weights)
    return layer.compile(
        TensorType(
            DType.float32, ("batch", "seq_len", cfg.hidden_size), device=DEVICE
        )
    )


SHAPES = [(1, 1), (1, 7), (2, 33), (1, 512)]


def check(run, shapes=SHAPES) -> list:
    rng = np.random.default_rng(0)
    results = []
    for batch, seq in shapes:
        x = rng.standard_normal((batch, seq, config.hidden_size)).astype(
            np.float32
        )
        actual = run(Tensor(x, device=DEVICE)).to_numpy()
        results.append(
            compare_outputs(
                actual, reference_layer(x), label=f"batch {batch}, seq {seq}"
            )
        )
    return results


correct = check(compiled_layer())
for result in correct:
    print(result)
assert all(r.passed for r in correct)
```

</details>

```text
batch 1, seq 1   mae 2.7e-07  max 2.9e-06  cos dist 2.4e-14  pass
batch 1, seq 7   mae 6.5e-07  max 1.7e-05  cos dist 1.9e-13  pass
batch 2, seq 33  mae 7.0e-07  max 2.1e-05  cos dist 1.9e-13  pass
batch 1, seq 512 mae 7.1e-07  max 3.8e-05  cos dist 2.0e-13  pass
```

Every shape passes, with maximum errors of `4e-5` or less and cosine
distances near zero.

## A test must be able to fail

A test that never fails proves nothing, so check that this one catches a real
bug. A plausible one is a config read wrong, with the base of RoPE
([Section 1.4](01_04_building_blocks.md)) at Llama 2's
`10000` instead of SmolLM2's `100000`:

```python
from dataclasses import replace

buggy = compiled_layer(cfg=replace(config, rope_theta=10000.0))
buggy_results = check(buggy)
for result in buggy_results:
    print(result)
assert not all(r.passed for r in buggy_results)
```

```text
batch 1, seq 1   mae 2.7e-07  max 2.9e-06  cos dist 2.4e-14  pass
batch 1, seq 7   mae 4.9e-04  max 6.6e-03  cos dist 1.2e-07  FAIL
batch 2, seq 33  mae 2.2e-03  max 1.3e-01  cos dist 2.3e-06  FAIL
batch 1, seq 512 mae 5.0e-03  max 2.3e-01  cos dist 1.1e-05  FAIL
```

<details class="code-fold">
<summary>Show the code <span class="code-fold-lines">45 lines</span></summary>

```python
ATOL, COS_THRESHOLD = 1e-3, 1e-5
fig = go.Figure(
    go.Scatter(
        x=[1e-7, ATOL, ATOL, 1e-7],
        y=[1e-15, 1e-15, COS_THRESHOLD, COS_THRESHOLD],
        fill="toself",
        fillcolor="rgba(27,175,122,0.15)",
        line=dict(color=viz.AQUA, width=1, dash="dot"),
        name="pass region (about)",
        hoverinfo="skip",
    )
)
for name, results, color in [
    ("correct layer", correct, viz.BLUE),
    ("rope_theta = 10000", buggy_results, viz.ORANGE),
]:
    fig.add_trace(
        go.Scatter(
            x=[r.max_abs_err for r in results],
            y=[max(r.cos_dist, 1e-15) for r in results],
            mode="markers+text",
            name=name,
            text=[
                r.label.replace("batch ", "b").replace(", seq ", "/s")
                for r in results
            ],
            textposition="middle right",
            marker=dict(
                color=color,
                size=11,
                symbol=["circle" if r.passed else "x" for r in results],
            ),
            hovertemplate="%{text}: max error %{x:.1e}, cos dist %{y:.1e}"
            + "<extra>"
            + name
            + "</extra>",
        )
    )
fig.update_layout(
    title="Each shape's max error against its cosine distance",
    height=440,
    xaxis=dict(title="max |error| (log scale)", type="log", dtick=1),
    yaxis=dict(title="cosine distance (log scale)", type="log", dtick=2),
)
viz.show(fig)
```

</details>

<div class="plotly-figure" data-src="generated/04_01_layer_tests_11_0.json"></div>

Each point is one shape, labeled batch/sequence length, and an `x` marks a
failure. The shaded corner is where both checks pass; its right edge is only
approximate, because `allclose` also allows a relative error of `1e-3` of each
expected value.

The single-token case still passes: at position 0, RoPE doesn't rotate at
all, so the bug can't show there. Every longer input fails. The maximum
error grows by two to four orders of magnitude and breaks the `allclose`
check. The cosine distance grows by six to eight orders of magnitude, but it
crosses its `1e-5` threshold only on the 512-token input. The two checks
catch different bugs, which is why the test uses both, and why the shape
list must include sequences, not just one token.

Where in the sequence does the bug show? Run the 512-token input through
both layers and plot the largest error at each position:

```python
x = (
    np.random.default_rng(1)
    .standard_normal((1, 512, config.hidden_size))
    .astype(np.float32)
)
expected = reference_layer(x)[0]
per_position = {
    name: np.abs(run(Tensor(x, device=DEVICE)).to_numpy()[0] - expected).max(-1)
    for name, run in [
        ("correct layer", compiled_layer()),
        ("rope_theta = 10000", buggy),
    ]
}
```

<details class="code-fold">
<summary>Show the code <span class="code-fold-lines">22 lines</span></summary>

```python
fig = go.Figure(
    [
        go.Scatter(
            x=np.arange(512),
            y=np.maximum(err, 1e-9),
            name=name,
            mode="lines",
            line_color=color,
            hovertemplate="position %{x}: %{y:.1e}<extra>" + name + "</extra>",
        )
        for (name, err), color in zip(
            per_position.items(), [viz.BLUE, viz.ORANGE]
        )
    ]
)
fig.update_layout(
    title="Largest error at each position of a 512-token input",
    height=400,
    xaxis_title="position",
    yaxis=dict(title="max |error| (log scale)", type="log", dtick=1),
)
viz.show(fig)
```

</details>

<div class="plotly-figure" data-src="generated/04_01_layer_tests_14_0.json"></div>

At position 0 the two layers agree, because RoPE doesn't rotate there.
Within a few positions the buggy layer's error is three to four orders of
magnitude larger, and it stays there for the whole sequence, while the
correct layer's error stays near `1e-5`. A test that only ran one token
would have passed both.

## As a pytest file

In a repository, these checks belong in a test file that runs on every
change. `%%writefile` saves a cell as a file. The test compiles the layer
once per module, as MAX's own tests do, and checks every shape:

<details class="code-fold">
<summary>Show the code <span class="code-fold-lines">46 lines</span></summary>

```python
%%writefile test_decoder_layer.py
import numpy as np
import pytest
import torch
from max.dtype import DType
from max.tensor import Tensor, default_device, default_dtype
from max.tensor.tensor import TensorType
from transformers import AutoConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRotaryEmbedding

from minillm.config import MODEL_ID, LlamaConfig
from minillm.device import pick_device
from minillm.model import DecoderLayer
from minillm.testing import compare_outputs
from minillm.weights import load_checkpoint, select

CONFIG = LlamaConfig.from_pretrained()


@pytest.fixture(scope="module")
def layers():
    weights = select(load_checkpoint(), "model.layers.0.")
    with default_device(pick_device()), default_dtype(DType.float32):
        ours = DecoderLayer(CONFIG)
    ours.load_state_dict(weights)
    compiled = ours.compile(
        TensorType(DType.float32, ("batch", "seq_len", CONFIG.hidden_size), device=pick_device())
    )
    hf_config = AutoConfig.from_pretrained(MODEL_ID)
    hf_config._attn_implementation = "eager"
    reference = LlamaDecoderLayer(hf_config, layer_idx=0).float().eval()
    reference.load_state_dict({k: torch.from_numpy(v.copy()) for k, v in weights.items()})
    return compiled, reference, LlamaRotaryEmbedding(hf_config)


@pytest.mark.parametrize("batch, seq", [(1, 1), (1, 7), (2, 33)])
def test_decoder_layer_matches_reference(layers, batch, seq):
    compiled, reference, rotary = layers
    x = np.random.default_rng(seq).standard_normal((batch, seq, CONFIG.hidden_size)).astype(np.float32)
    hidden, positions = torch.from_numpy(x), torch.arange(seq).expand(batch, seq)
    mask = torch.full((seq, seq), float("-inf")).triu(1)[None, None]
    with torch.no_grad():
        expected = reference(hidden, attention_mask=mask, position_ids=positions,
                             position_embeddings=rotary(hidden, positions)).numpy()
    result = compare_outputs(compiled(Tensor(x, device=pick_device())).to_numpy(), expected)
    assert result.passed, result
```

</details>

```text
Overwriting test_decoder_layer.py
```

```python
!python -m pytest -q -p no:cacheprovider test_decoder_layer.py 2>&1 | tail -n 3
```

```text
...                                                                      [100%]
3 passed in 60.67s (0:01:00)
```

MAX's own layer tests follow the same shape, with one more step: each harness
also *benchmarks* the layer on the same shapes, so correctness and speed are
measured together.
[Section 4.3](04_03_kernel_benchmarks.md) covers
benchmarking.

## Exercises

1. Add a `bfloat16` run: cast the input and the weights, and compare with the
   `float32` reference. Which tolerances do you need, and which metric moves
   most?
2. Test `CachedAttention` from
   [Section 1.8](01_08_kv_cache.md): compare a prefill
   of 8 tokens followed by one decode step with a single 9-token pass.
3. Plant a different bug, such as swapping `gate_proj` and `up_proj`. Which
   metric catches it first?

## Next

[Section 4.2](04_02_logit_verification.md) tests the
whole model: its logits on real prompts, against the same reference.

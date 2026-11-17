# 7.1 Serve your architecture

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/modular/max-llm-book/blob/main/notebooks/07_01_serve_your_model.ipynb)
[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/modular/max-llm-book/blob/main/notebooks/07_01_serve_your_model.ipynb)
[![View on GitHub](https://img.shields.io/badge/GitHub-notebook-black?logo=github)](https://github.com/modular/max-llm-book/blob/main/notebooks/07_01_serve_your_model.ipynb)

In [Section 1.1](01_01_run_and_serve.md) you served
SmolLM2 with MAX's own implementation. This section serves *yours*: the
`LlamaForCausalLM` you wrote in
[Section 1.6](01_06_full_model.md), behind the same
`AutoPipeline` API and the same OpenAI-compatible server. MAX loads model code
from outside its own package through *custom architectures*: a Python package
that describes the model, which `max serve` imports with
`--custom-architectures`.

<details class="code-fold">
<summary>Show the setup <span class="code-fold-lines">14 lines</span></summary>

```python
import logging
import time
import warnings

from max.pipelines import AutoPipeline
from max.pipelines.lib.arch_lookup import import_custom_architectures
from openai import OpenAI

from minillm.config import MODEL_ID
from minillm.server import start_max_serve

logging.getLogger("max").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="The eager interpreter failed")
PROMPT = ["The capital of France is"]
```

</details>

First, the reference: SmolLM2 as MAX implements it, loaded with
`AutoPipeline`.

```python
reference = AutoPipeline(MODEL_ID, quantization_encoding="float32")
expected = reference(PROMPT, max_new_tokens=16, top_k=1)
reference.close()
expected
```

```text
['The capital of France is Paris. Paris is the largest city in France and the capital of the French department']
```

## A custom architecture package

A MAX architecture has three parts:

- A config class tells MAX the sizes it needs to plan memory before it loads
  anything: the number of layers, key/value heads, and head size. It also
  declares the supported encodings; this model computes in `float32`.
- A pipeline model loads the weights and runs the model. MAX gives it the
  checkpoint's weights and, at each step, the request's tokens, and it returns
  the next-token logits.
- A `SupportedArchitecture` ties them together under the name in the
  checkpoint's `config.json`, here `LlamaForCausalLM`, along with the tokenizer
  and the weight format. `ARCHITECTURES` lists it for `--custom-architectures`
  to find.

To keep the package short, this pipeline model reruns the whole sequence at each
step and serves one request at a time (`max_batch_size` is 1), like the
generation loop in [Section 1.7](01_07_sampling.md).
MAX's own architectures use its paged KV cache and continuous batching instead:
they implement a decoder module that reads keys and values from the cache. MAX's
OLMo 3 architecture is a compact example.

The package is one module, `minillm/serve/__init__.py`, which the cells below
write part by part. The first imports what the package uses from MAX:

<details class="code-fold">
<summary>Show the imports <span class="code-fold-lines">41 lines</span></summary>

```python
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from max.driver import Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef
from max.graph.weights import Weights, WeightsAdapter, WeightsFormat
from max.nn.kv_cache import KVCacheInputs, KVCacheParams
from max.nn.outputs import ReturnHiddenStates, ReturnLogits
from max.pipelines.context import TextContext
from max.pipelines.lib import (
    KVCacheConfig,
    MAXModelConfig,
    MemoryPlan,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
    PipelineModelWithKVCache,
    SupportedArchitecture,
    SupportedEncoding,
    TextTokenizer,
)
from max.pipelines.lib.interfaces.arch_config import (
    ArchConfigWithAttentionKVCache,
)
from max.pipelines.lib.utils import (
    parse_state_dict_from_weights,
    upper_bounded_default,
)
from max.pipelines.modeling.types import PipelineTask
from max.tensor import Tensor
from max.tensor.tensor import TensorType
from transformers import AutoConfig

from minillm.config import LlamaConfig
from minillm.model import load_model
```

</details>

<small>Saved to `minillm/serve/__init__.py`.</small>

The config class reads the sizes from the checkpoint's Hugging Face config:

<details class="code-fold">
<summary>Show <code>MinillmArchConfig</code> <span class="code-fold-lines">29 lines</span></summary>

```python
@dataclass
class MinillmArchConfig(ArchConfigWithAttentionKVCache):
    DEFAULT_ENCODING: ClassVar[SupportedEncoding] = "float32"
    SUPPORTED_ENCODINGS: ClassVar[set[SupportedEncoding]] = {"float32"}

    @property
    def num_key_value_heads(self) -> int:
        return self.huggingface_config.num_key_value_heads

    @property
    def head_dim(self) -> int:
        hf = self.huggingface_config
        return (
            getattr(hf, "head_dim", None)
            or hf.hidden_size // hf.num_attention_heads
        )

    @property
    def num_layers(self) -> int:
        return self.huggingface_config.num_hidden_layers

    @classmethod
    def calculate_max_seq_len(
        cls, huggingface_config: AutoConfig, model_config: MAXModelConfig
    ) -> int:
        return upper_bounded_default(
            upper_bound=huggingface_config.max_position_embeddings,
            default=model_config.max_length,
        )
```

</details>

<small>Saved to `minillm/serve/__init__.py`.</small>

The pipeline model's main job is to build your model. `build_model` turns the
checkpoint's config into the `LlamaConfig` from
[Section 1.3](01_03_tokens_and_weights.md), casts the
checkpoint's `bfloat16` weights to `float32`, builds your `LlamaForCausalLM` with
`load_model` from [Section 1.6](01_06_full_model.md),
and compiles it for any sequence length:

```python
def build_model(
    hf: AutoConfig,
    pipeline_config: PipelineConfig,
    weights: Weights,
    adapter: WeightsAdapter | None,
    device: Device,
) -> Callable[[Tensor], Tensor]:
    """Builds and compiles minillm's ``LlamaForCausalLM`` from a checkpoint."""
    config = LlamaConfig(
        **{
            k: getattr(hf, k)
            for k in LlamaConfig.__dataclass_fields__
            if hasattr(hf, k)
        }
    )
    state = parse_state_dict_from_weights(pipeline_config, weights, adapter)
    state = {name: w.astype(DType.float32) for name, w in state.items()}
    model = load_model(config, state, device)
    return model.compile(
        TensorType(DType.int64, ("batch", "seq_len"), device=device)
    )
```

<small>Saved to `minillm/serve/__init__.py`.</small>

MAX constructs the pipeline model with a long list of arguments, and asks it
how to size the KV cache. The base class below passes both through and calls
`build_model`:

<details class="code-fold">
<summary>Show <code>TokenInputs</code>, <code>PassThroughPipelineModel</code> <span class="code-fold-lines">66 lines</span></summary>

```python
@dataclass
class TokenInputs(ModelInputs):
    tokens: Buffer

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        return (self.tokens,)


class PassThroughPipelineModel(PipelineModelWithKVCache[TextContext]):
    """Forwards MAX's constructor arguments and sizes the KV cache from the
    checkpoint's config, then builds the model with ``build_model``."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        devices: list[Device],
        kv_cache_config: KVCacheConfig,
        weights: Weights,
        adapter: WeightsAdapter | None,
        return_logits: ReturnLogits,
        *,
        memory_plan: MemoryPlan,
        return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE,
        max_batch_size: int = 1,
    ) -> None:
        super().__init__(
            pipeline_config=pipeline_config,
            session=session,
            devices=devices,
            kv_cache_config=kv_cache_config,
            weights=weights,
            adapter=adapter,
            return_logits=return_logits,
            return_hidden_states=return_hidden_states,
            max_batch_size=max_batch_size,
            memory_plan=memory_plan,
        )
        self.model = build_model(
            self.huggingface_config,
            self.pipeline_config,
            weights,
            adapter,
            self.devices[0],
        )

    @classmethod
    def get_kv_params(
        cls,
        huggingface_config: Any,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        hf = huggingface_config
        return kv_cache_config.to_params(
            dtype=cache_dtype,
            n_kv_heads=hf.num_key_value_heads,
            head_dim=getattr(hf, "head_dim", None)
            or hf.hidden_size // hf.num_attention_heads,
            num_layers=hf.num_hidden_layers,
            devices=devices,
            data_parallel_degree=pipeline_config.model.data_parallel_degree,
        )
```

</details>

<small>Saved to `minillm/serve/__init__.py`.</small>

That leaves the two methods MAX calls at every step.
`prepare_initial_token_inputs` turns the request's context into model inputs:
here all of its tokens, since the model keeps no cache. `execute` runs the
compiled model and returns the last position's logits:

```python
class MinillmPipelineModel(PassThroughPipelineModel):
    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputs[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> TokenInputs:
        ids = replica_batches[0][0].tokens.all
        tokens = np.array([ids], dtype=np.int64)
        inputs = TokenInputs(Buffer.from_numpy(tokens).to(self.devices[0]))
        inputs.kv_cache_inputs = kv_cache_inputs
        return inputs

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        assert isinstance(model_inputs, TokenInputs)
        logits = self.model(Tensor.from_dlpack(model_inputs.tokens))
        last = Buffer.from_numpy(
            np.ascontiguousarray(logits.to_numpy()[0, -1:, :])
        ).to(self.devices[0])
        return ModelOutputs(logits=last, next_token_logits=last)
```

<small>Saved to `minillm/serve/__init__.py`.</small>

Finally, `SupportedArchitecture` registers the parts under the checkpoint's
architecture name. `required_arguments` makes MAX serve one request at a time
and turns off prefix caching, which this model can't use:

```python
def _identity_adapter(state_dict: dict[str, Weights], **_: object) -> dict:
    return {name: w.data() for name, w in state_dict.items()}


minillm_arch = SupportedArchitecture(
    name="LlamaForCausalLM",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=["HuggingFaceTB/SmolLM2-135M-Instruct"],
    default_weights_format=WeightsFormat.safetensors,
    default_encoding=MinillmArchConfig.DEFAULT_ENCODING,
    supported_encodings=MinillmArchConfig.SUPPORTED_ENCODINGS,
    pipeline_model=MinillmPipelineModel,
    tokenizer=TextTokenizer,
    context_type=TextContext,
    multi_gpu_supported=False,
    weight_adapters={WeightsFormat.safetensors: _identity_adapter},
    config=MinillmArchConfig,
    required_arguments={"enable_prefix_caching": False, "max_batch_size": 1},
    supports_overlap_scheduler=False,
    supports_device_graph_capture=False,
)
ARCHITECTURES = [minillm_arch]
```

<small>Saved to `minillm/serve/__init__.py`.</small>

## Use it in Python

`import_custom_architectures` registers the package's architectures, as
`--custom-architectures` does for the CLI. A registered architecture with
the same name as one of MAX's replaces it in this process. When an
architecture has a newer module-based implementation, MAX prefers that one,
so `prefer_module_v3=False` selects the registered `LlamaForCausalLM`:

```python
import_custom_architectures(["minillm.serve"])
ours = AutoPipeline(
    MODEL_ID, quantization_encoding="float32", prefer_module_v3=False
)
print("pipeline model:", type(ours.model.pipeline._pipeline_model).__name__)
result = ours(PROMPT, max_new_tokens=16, top_k=1)
ours.close()
print(result)
print("same as MAX's implementation:", result == expected)
assert result == expected
```

```text
pipeline model: MinillmPipelineModel
['The capital of France is Paris. Paris is the largest city in France and the capital of the French department']
same as MAX's implementation: True
```

The pipeline model is yours, and its greedy output matches MAX's
implementation token for token.

## Serve it

The same package serves over HTTP. `--custom-architectures minillm.serve`
imports it, and `--no-prefer-module-v3` selects it over MAX's own
implementation:

```python
start = time.perf_counter()
server = start_max_serve(
    MODEL_ID,
    "--quantization-encoding",
    "float32",
    "--custom-architectures",
    "minillm.serve",
    "--no-prefer-module-v3",
)
print(f"server ready in {time.perf_counter() - start:.0f} s")

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
completion = client.completions.create(
    model=MODEL_ID, prompt=PROMPT[0], max_tokens=16, temperature=0
)
print(PROMPT[0] + completion.choices[0].text)
assert PROMPT[0] + completion.choices[0].text == expected[0]
server.terminate()
_ = server.wait()
```

```text
server ready in 18 s
The capital of France is Paris. Paris is the largest city in France and the capital of the French department
```

Your model now answers an OpenAI client exactly as MAX's implementation does.
The handbook's
[OpenAI-compatible API](https://handbook.modular.com/model-interaction/openai-compatible-api/)
page covers the request format. MAX's guide to
[serving custom model architectures](https://max.modular.com/develop/serve-custom-model-architectures)
covers the production version: a decoder module on MAX's paged KV cache, with
batching and GPU kernels.

## Exercises

1. Register `Qwen3ForCausalLM` from
   [Section 6.1](06_01_qwen3.md) the same way, and
   serve `Qwen/Qwen3-0.6B` through it. Which parts of the package depend on the
   model class, and which work unchanged?
2. Make the pipeline model serve more than one request per step: read every
   context in `replica_batches[0]`, pad them as `run_step` from
   [Section 5.1](05_01_request_lifecycle.md) does, and
   return one row of logits per request.
3. Replace the full recompute with `KVCacheDecoder` from
   [Section 1.8](01_08_kv_cache.md), keeping one cache
   per request ID. What happens when a request finishes?

## Next

[Section 7.2](07_02_benchmark_serving.md) measures the
server under load with `max benchmark`.

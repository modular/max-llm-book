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

You register the model twice. The first registration runs your model
unchanged and recomputes the whole sequence at every step. The second wraps
your layers in a *decoder module* that reads keys and values from MAX's paged
KV cache, which is how MAX's own architectures serve.

<details class="code-fold">
<summary>Show the setup <span class="code-fold-lines">13 lines</span></summary>

```python
import logging
import time
import warnings

from max.pipelines import AutoPipeline
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

To keep this first version short, the pipeline model reruns the whole sequence
at each step and serves one request at a time (`max_batch_size` is 1), like the
generation loop in [Section 1.7](01_07_sampling.md).
[A decoder on the paged KV cache](#a-decoder-on-the-paged-kv-cache) below
removes both limits.

The package is one module, `minillm/serve/__init__.py`, which the cells below
write part by part. The first imports what the package uses from MAX:

<details class="code-fold">
<summary>Show the imports <span class="code-fold-lines">43 lines</span></summary>

```python
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from max import functional as F
from max.driver import CPU, Buffer, Device
from max.dtype import DType
from max.graph import TensorType
from max.graph.weights import Weights, WeightsAdapter, WeightsFormat
from max.nn import Embedding, Linear, Module, ModuleList, RMSNorm
from max.nn.attention_mask import MHAMaskVariant
from max.nn.common_layers.kv_cache import per_device_caches
from max.nn.kernels import flash_attention_ragged, rope_split_store_ragged
from max.nn.kv_cache import KVCacheInputs, KVCacheParams, PagedCacheValues
from max.pipelines import upper_bounded_default
from max.pipelines.context import TextContext
from max.pipelines.lib import (
    GraphRole,
    MAXModelConfig,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
    PipelineModelWithKVCache,
    SupportedArchitecture,
    SupportedEncoding,
    TextTokenizer,
)
from max.pipelines.lib.interfaces import (
    ArchConfigWithAttentionKVCache,
    ArchConfigWithStoredKVParams,
)
from max.pipelines.lib.role_contracts import DecoderInputs, DecoderOutputs
from max.pipelines.lib.utils import parse_state_dict_from_weights
from max.pipelines.modeling.types import PipelineTask
from max.tensor import Tensor
from transformers import AutoConfig

from minillm.config import LlamaConfig
from minillm.layers import MLP, rope_cos_sin
from minillm.model import load_model
```

</details>

<small>Saved to `minillm/serve/__init__.py`.</small>

The config class reads the sizes from the checkpoint's Hugging Face config.
`ArchConfigWithStoredKVParams` adds `construct_kv_params`, which MAX calls to
size the KV cache from those sizes. `llama_config` converts the same config to
the `LlamaConfig` from
[Section 1.3](01_03_tokens_and_weights.md):

<details class="code-fold">
<summary>Show <code>MinillmArchConfig</code>, <code>llama_config</code> <span class="code-fold-lines">38 lines</span></summary>

```python
@dataclass
class MinillmArchConfig(
    ArchConfigWithAttentionKVCache, ArchConfigWithStoredKVParams
):
    DEFAULT_ENCODING: ClassVar[SupportedEncoding] = "float32"
    SUPPORTED_ENCODINGS: ClassVar[set[SupportedEncoding]] = {"float32"}

    @property
    def num_key_value_heads(self) -> int:
        return self.huggingface_config.num_key_value_heads

    @property
    def head_dim(self) -> int:
        return self.get_head_dim(self.huggingface_config)

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


def llama_config(hf: AutoConfig) -> LlamaConfig:
    """Returns the checkpoint's sizes as a ``LlamaConfig``."""
    return LlamaConfig(
        **{
            k: getattr(hf, k)
            for k in LlamaConfig.__dataclass_fields__
            if hasattr(hf, k)
        }
    )
```

</details>

<small>Saved to `minillm/serve/__init__.py`.</small>

The pipeline model's main job is to build your model. `build_model` casts the
checkpoint's `bfloat16` weights to `float32`, builds your `LlamaForCausalLM` with
`load_model` from [Section 1.6](01_06_full_model.md),
and compiles it for any sequence length:

```python
def build_model(
    config: LlamaConfig,
    pipeline_config: PipelineConfig,
    weights: Weights,
    adapter: WeightsAdapter | None,
    device: Device,
) -> Callable[[Tensor], Tensor]:
    """Builds and compiles minillm's ``LlamaForCausalLM`` from a checkpoint."""
    state = parse_state_dict_from_weights(pipeline_config, weights, adapter)
    state = {name: w.astype(DType.float32) for name, w in state.items()}
    model = load_model(config, state, device)
    return model.compile(
        TensorType(DType.int64, ("batch", "seq_len"), device=device)
    )
```

<small>Saved to `minillm/serve/__init__.py`.</small>

MAX constructs the pipeline model with a long list of arguments, which the
constructor passes through before it calls `build_model`. `model_config_cls`
names the config class, which MAX asks to size the KV cache.
`prepare_initial_token_inputs` turns the request's context into model inputs:
here all of its tokens, since the model keeps no cache. `execute` runs the
compiled model and returns the last position's logits:

<details class="code-fold">
<summary>Show <code>TokenInputs</code>, <code>MinillmPipelineModel</code> <span class="code-fold-lines">41 lines</span></summary>

```python
@dataclass
class TokenInputs(ModelInputs):
    tokens: Buffer

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        return (self.tokens,)


class MinillmPipelineModel(PipelineModelWithKVCache[TextContext]):
    model_config_cls: ClassVar[type[MinillmArchConfig]] = MinillmArchConfig

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.model = build_model(
            llama_config(self.huggingface_config),
            self.pipeline_config,
            self.weights,
            self.adapter,
            self.devices[0],
        )

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

</details>

<small>Saved to `minillm/serve/__init__.py`.</small>

Finally, `SupportedArchitecture` registers the parts under the checkpoint's
architecture name. `required_arguments` makes MAX serve one request at a time
and turns off prefix caching, which this model can't use:

```python
def tie_embeddings(state_dict: dict[str, Weights], **_: object) -> dict:
    """Loads the checkpoint as is, with the LM head sharing the embedding."""
    weights = {name: w.data() for name, w in state_dict.items()}
    weights.setdefault("lm_head.weight", weights["model.embed_tokens.weight"])
    return weights


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
    weight_adapters={WeightsFormat.safetensors: tie_embeddings},
    config=MinillmArchConfig,
    required_arguments={"enable_prefix_caching": False, "max_batch_size": 1},
    supports_overlap_scheduler=False,
    supports_device_graph_capture=False,
)
```

<small>Saved to `minillm/serve/__init__.py`.</small>

## Use it in Python

`custom_architectures` imports the package and registers its architectures,
as `--custom-architectures` does for the CLI. A registered architecture with
the same name as one of MAX's replaces it in this process. MAX registers
module-based implementations under the name with a `_ModuleV3` suffix and
prefers them, so `prefer_module_v3=False` selects the `LlamaForCausalLM`
registered above:

```python
ours = AutoPipeline(
    MODEL_ID,
    quantization_encoding="float32",
    custom_architectures=["minillm.serve"],
    prefer_module_v3=False,
)
result = ours(PROMPT, max_new_tokens=16, top_k=1)
ours.close()
print(result)
print("same as MAX's implementation:", result == expected)
assert result == expected
```

```text
['The capital of France is Paris. Paris is the largest city in France and the capital of the French department']
same as MAX's implementation: True
```

The pipeline model is yours, and its greedy output matches MAX's
implementation token for token.

## A decoder on the paged KV cache

The pipeline model above recomputes every token at every step and runs one
request at a time. MAX's own architectures don't write a pipeline model.
They register a *decoder module* for the `DECODER` role, and MAX builds the
pipeline model around it. That pipeline model brings what
[Chapter 5](ch05.md) built by hand: continuous
batching, a paged KV cache, and prefix caching.

At each step, the decoder's `forward` receives a `DecoderInputs`:

- `tokens`: the new tokens of every request in the batch, concatenated into
  one *ragged* sequence. A request's prompt arrives whole at its first step,
  and then one token per step.
- `input_row_offsets`: where each request starts in `tokens`, plus the total
  length at the end.
- `kv`: the paged KV cache. `per_device_caches` unpacks it into one
  `PagedCacheValues` per device.

Your embedding, norms, and MLP work on ragged tokens unchanged, since they
treat each token on its own. Only attention needs the cache, and two MAX
kernels supply it. `rope_split_store_ragged` splits the fused Q/K/V
projection, rotates the queries and keys, and writes the keys and values
into the cache's pages. `flash_attention_ragged` then attends each new query
to its own request's cached keys, with a causal mask. The kernels take the
RoPE table packed as (cos, sin) pairs, which `rope_table` builds from your
`rope_cos_sin` from
[Section 1.4](01_04_building_blocks.md):

<details class="code-fold">
<summary>Show <code>rope_table</code>, <code>PagedAttention</code> <span class="code-fold-lines">59 lines</span></summary>

```python
def rope_table(config: LlamaConfig, length: int, device: Device) -> Tensor:
    """Returns the RoPE table as ``[length, head_dim]`` (cos, sin) pairs."""
    cos, sin = rope_cos_sin(
        config.head_dim, config.rope_theta, 0, length, device
    )
    half = config.head_dim // 2
    pairs = F.stack([cos[:, :half], sin[:, :half]], axis=-1)
    return pairs.reshape([length, config.head_dim])


class PagedAttention(Module):
    """Grouped-query attention that reads keys and values from the cache."""

    def __init__(
        self, config: LlamaConfig, kv_params: KVCacheParams, layer_idx: int
    ) -> None:
        q_dim = config.num_attention_heads * config.head_dim
        kv_dim = config.num_key_value_heads * config.head_dim
        self.q_proj = Linear(config.hidden_size, q_dim, bias=False)
        self.k_proj = Linear(config.hidden_size, kv_dim, bias=False)
        self.v_proj = Linear(config.hidden_size, kv_dim, bias=False)
        self.o_proj = Linear(q_dim, config.hidden_size, bias=False)
        self.n_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.kv_params = kv_params
        self.layer_idx = layer_idx

    def forward(
        self,
        x: Tensor,
        kv: PagedCacheValues,
        rope: Tensor,
        input_row_offsets: Tensor,
    ) -> Tensor:
        qkv = F.concat(
            [self.q_proj(x), self.k_proj(x), self.v_proj(x)], axis=-1
        )
        # The kernels read the layer index on the host.
        layer = F.constant(self.layer_idx, DType.uint32, device=CPU())
        q = rope_split_store_ragged(
            kv_params=self.kv_params,
            qkv=qkv,
            input_row_offsets=input_row_offsets,
            freqs_cis=rope,
            kv_collection=kv,
            layer_idx=layer,
            n_heads=self.n_heads,
            interleaved=False,
        )
        out = flash_attention_ragged(
            self.kv_params,
            input=q.reshape([-1, self.n_heads, self.head_dim]),
            input_row_offsets=input_row_offsets,
            kv_collection=kv,
            layer_idx=layer,
            mask_variant=MHAMaskVariant.CAUSAL_MASK,
            scale=self.head_dim**-0.5,
        )
        return self.o_proj(out.reshape([-1, self.n_heads * self.head_dim]))
```

</details>

<small>Saved to `minillm/serve/__init__.py`.</small>

The layer, the model, and the decoder mirror `DecoderLayer`, `LlamaModel`,
and `LlamaForCausalLM` from
[Section 1.6](01_06_full_model.md), with the same
attribute names, so the checkpoint loads into them unchanged. MAX constructs
the decoder as `MinillmDecoder(config)` from the config class. The decoder
returns the logits of each request's last token, the one its next token
follows:

<details class="code-fold">
<summary>Show <code>PagedDecoderLayer</code>, <code>PagedLlamaModel</code>, <code>MinillmDecoder</code> <span class="code-fold-lines">55 lines</span></summary>

```python
class PagedDecoderLayer(Module):
    def __init__(
        self, config: LlamaConfig, kv_params: KVCacheParams, layer_idx: int
    ) -> None:
        eps = config.rms_norm_eps
        self.input_layernorm = RMSNorm(config.hidden_size, eps=eps)
        self.self_attn = PagedAttention(config, kv_params, layer_idx)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=eps)
        self.mlp = MLP(config)

    def forward(
        self,
        x: Tensor,
        kv: PagedCacheValues,
        rope: Tensor,
        input_row_offsets: Tensor,
    ) -> Tensor:
        h = self.input_layernorm(x)
        x = x + self.self_attn(h, kv, rope, input_row_offsets)
        return x + self.mlp(self.post_attention_layernorm(x))


class PagedLlamaModel(Module):
    def __init__(self, config: LlamaConfig, kv_params: KVCacheParams) -> None:
        self.embed_tokens = Embedding(config.vocab_size, dim=config.hidden_size)
        self.layers = ModuleList(
            [
                PagedDecoderLayer(config, kv_params, i)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class MinillmDecoder(Module[[DecoderInputs], DecoderOutputs]):
    def __init__(self, arch_config: MinillmArchConfig) -> None:
        self.config = llama_config(arch_config.huggingface_config)
        self.max_seq_len = arch_config.max_seq_len
        self.kv_params = arch_config.get_kv_params()
        self.model = PagedLlamaModel(self.config, self.kv_params)
        self.lm_head = Linear(
            self.config.hidden_size, self.config.vocab_size, bias=False
        )

    def forward(self, inputs: DecoderInputs) -> DecoderOutputs:
        (kv,) = per_device_caches(self.kv_params, inputs.kv)
        offsets = inputs.input_row_offsets
        rope = rope_table(self.config, self.max_seq_len, inputs.tokens.device)
        x = self.model.embed_tokens(inputs.tokens)
        for layer in self.model.layers:
            x = layer(x, kv, rope, offsets)
        last = F.gather(x, offsets[1:] - 1, axis=0)
        return DecoderOutputs(
            next_token_logits=self.lm_head(self.model.norm(last))
        )
```

</details>

<small>Saved to `minillm/serve/__init__.py`.</small>

The registration names the decoder under `modules` instead of a pipeline
model. Its name, `LlamaForCausalLM_ModuleV3`, replaces MAX's own module-based
Llama, so MAX selects it by default:

```python
minillm_decoder_arch = SupportedArchitecture(
    name="LlamaForCausalLM_ModuleV3",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=["HuggingFaceTB/SmolLM2-135M-Instruct"],
    default_weights_format=WeightsFormat.safetensors,
    default_encoding=MinillmArchConfig.DEFAULT_ENCODING,
    supported_encodings=MinillmArchConfig.SUPPORTED_ENCODINGS,
    modules={GraphRole.DECODER: MinillmDecoder},
    tokenizer=TextTokenizer,
    context_type=TextContext,
    weight_adapters={WeightsFormat.safetensors: tie_embeddings},
    config=MinillmArchConfig,
    supports_overlap_scheduler=False,
    supports_device_graph_capture=False,
)
ARCHITECTURES = [minillm_arch, minillm_decoder_arch]
```

<small>Saved to `minillm/serve/__init__.py`.</small>

This time MAX batches requests. Two prompts run in one batch, and the first
still matches MAX's implementation:

```python
paged = AutoPipeline(
    MODEL_ID,
    quantization_encoding="float32",
    custom_architectures=["minillm.serve"],
    max_batch_size=2,
)
batch = paged(PROMPT + ["Once upon a time"], max_new_tokens=16, top_k=1)
paged.close()
for text in batch:
    print(text)
print("same as MAX's implementation:", batch[0] == expected[0])
assert batch[0] == expected[0]
```

```text
The capital of France is Paris. Paris is the largest city in France and the capital of the French department
Once upon a time, I was a young man, a young woman, and a young man again
same as MAX's implementation: True
```

## Serve it

The same package serves over HTTP. `--custom-architectures minillm.serve`
imports it, and MAX serves the decoder:

```python
start = time.perf_counter()
server = start_max_serve(
    MODEL_ID,
    "--quantization-encoding",
    "float32",
    "--custom-architectures",
    "minillm.serve",
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
server ready in 20 s
The capital of France is Paris. Paris is the largest city in France and the capital of the French department
```

Your model now answers an OpenAI client exactly as MAX's implementation does,
on MAX's paged KV cache and scheduler. The handbook's
[OpenAI-compatible API](https://handbook.modular.com/model-interaction/openai-compatible-api/)
page covers the request format. MAX's guide to
[serving custom model architectures](https://max.modular.com/develop/serve-custom-model-architectures)
covers the rest of a production architecture, such as weight adapters for
other checkpoint formats and multi-GPU serving.

## Exercises

1. Register `Qwen3ForCausalLM` from
   [Section 6.1](06_01_qwen3.md) as a decoder, and
   serve `Qwen/Qwen3-0.6B` through it. `rope_split_store_ragged` takes
   `q_norm_weight`, `k_norm_weight`, and `rms_norm_eps` for Qwen3's per-head
   norms. Which parts of the package depend on the model class, and which
   work unchanged?
2. Serve each registration in turn, `--no-prefer-module-v3` selecting the
   first, and measure both with `max benchmark` from
   [Section 7.2](07_02_benchmark_serving.md). How
   does throughput change as you raise `--max-concurrency`?
3. `rope_table` rebuilds the RoPE table at every step. Build it once in
   `MinillmDecoder.__init__` instead. Why does MAX then report the table as a
   weight missing from the checkpoint, and how could the weight adapter
   supply it?

## Next

[Section 7.2](07_02_benchmark_serving.md) measures the
server under load with `max benchmark`.

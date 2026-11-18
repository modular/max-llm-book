# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Generated from 07_01_serve_your_model.ipynb by tools/nb2book.py.

Edit the notebook, then run ``pixi run export``.
"""

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

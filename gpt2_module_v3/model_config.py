# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from __future__ import annotations

from dataclasses import dataclass

from max.dtype import DType
from max.graph import DeviceRef
from max.graph.weights import WeightData, WeightsFormat, weights_format
from max.nn import ReturnLogits
from max.nn.kv_cache import KVCacheParams
from max.pipelines.lib import (
    KVCacheConfig,
    MAXModelConfig,
    MAXModelConfigBase,
    PipelineConfig,
)
from transformers import AutoConfig


@dataclass
class GPT2ConfigBase(MAXModelConfigBase):
    """Base configuration for GPT-2 models."""

    vocab_size: int
    """Vocabulary size of the GPT-2 model."""

    n_positions: int
    """Maximum sequence length the model can handle."""

    n_embd: int
    """Dimension of the hidden representations."""

    n_layer: int
    """Number of hidden layers in the Transformer decoder."""

    n_head: int
    """Number of attention heads for each attention layer."""

    n_inner: int | None
    """Dimension of the MLP representations. If None, defaults to 4 * n_embd."""

    layer_norm_epsilon: float
    """The epsilon used by the layer normalization layers."""

    # MAX-specific config parameters
    dtype: DType
    """DType of the model weights and input."""

    devices: list[DeviceRef]
    """Devices to run the model with."""

    return_logits: ReturnLogits
    """Whether to return the last token, all logits, or a variable number of logits."""

    kv_params: KVCacheParams
    """KV cache parameters."""


@dataclass
class GPT2Config(MAXModelConfig, GPT2ConfigBase):
    """Represents the complete MAX Engine configuration for GPT-2 models."""

    @staticmethod
    def get_kv_params(
        huggingface_config: AutoConfig,
        n_devices: int,
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        """Constructs the KV cache parameters from configuration objects."""
        return KVCacheParams(
            dtype=cache_dtype,
            num_layers=GPT2Config.get_num_layers(huggingface_config),
            n_kv_heads=huggingface_config.n_head,
            head_dim=huggingface_config.n_embd // huggingface_config.n_head,
            page_size=kv_cache_config.kv_cache_page_size,
            cache_strategy=kv_cache_config.cache_strategy,
            enable_prefix_caching=kv_cache_config.enable_prefix_caching,
            enable_kvcache_swapping_to_host=kv_cache_config.enable_kvcache_swapping_to_host,
            host_kvcache_swap_space_gb=kv_cache_config.host_kvcache_swap_space_gb,
            n_devices=n_devices,
        )

    @staticmethod
    def get_num_layers(huggingface_config: AutoConfig) -> int:
        """Retrieves the number of hidden layers from the HuggingFace configuration."""
        return huggingface_config.n_layer

    @staticmethod
    def calculate_max_seq_len(
        pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        """Calculates the maximum sequence length for the model."""
        max_seq_len = pipeline_config.max_length
        if max_seq_len:
            return max_seq_len
        return huggingface_config.n_positions

    @staticmethod
    def generate(
        pipeline_config: PipelineConfig,
        huggingface_config: AutoConfig,
        state_dict: dict[str, WeightData],
        dtype: DType,
        n_devices: int,
        cache_dtype: DType,
        kv_cache_config: KVCacheConfig,
        return_logits: ReturnLogits,
    ) -> GPT2Config:
        """Generates a GPT2Config instance from various configuration sources."""
        device_refs = [
            DeviceRef(spec.device_type, spec.id)
            for spec in pipeline_config.model_config.device_specs
        ]

        return GPT2Config(
            vocab_size=huggingface_config.vocab_size,
            n_positions=huggingface_config.n_positions,
            n_embd=huggingface_config.n_embd,
            n_layer=huggingface_config.n_layer,
            n_head=huggingface_config.n_head,
            n_inner=getattr(huggingface_config, "n_inner", None),
            layer_norm_epsilon=huggingface_config.layer_norm_epsilon,
            dtype=dtype,
            devices=device_refs,
            return_logits=return_logits,
            kv_params=GPT2Config.get_kv_params(
                huggingface_config=huggingface_config,
                n_devices=n_devices,
                kv_cache_config=kv_cache_config,
                cache_dtype=cache_dtype,
            ),
        )

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
# gpt2_arch/model_config.py
"""GPT-2 architecture configuration for max serve."""

# ANCHOR: book
from __future__ import annotations

from dataclasses import dataclass

from max.pipelines.lib import MAXModelConfig, PipelineConfig
from max.pipelines.lib.interfaces.arch_config import (
    ArchConfigWithAttentionKVCache,
)
from max.pipelines.lib.utils import upper_bounded_default
from transformers import AutoConfig


@dataclass
class GPT2ArchConfig(ArchConfigWithAttentionKVCache):
    @property
    def num_key_value_heads(self) -> int:
        """GPT-2 uses plain MHA: n_kv_heads == n_head."""
        return self.huggingface_config.n_head  # type: ignore[union-attr]

    @property
    def head_dim(self) -> int:
        hf = self.huggingface_config
        return hf.n_embd // hf.n_head  # type: ignore[union-attr]

    @property
    def num_layers(self) -> int:
        return self.huggingface_config.n_layer  # type: ignore[union-attr]

    @classmethod
    def calculate_max_seq_len(
        cls,
        pipeline_config: PipelineConfig,
        huggingface_config: AutoConfig,
        model_config: MAXModelConfig | None = None,
    ) -> int:
        """The user's max_length, bounded by GPT-2's position count."""
        model_config = model_config or pipeline_config.model
        return upper_bounded_default(
            upper_bound=huggingface_config.n_positions,
            default=model_config.max_length,
        )


# ANCHOR_END: book

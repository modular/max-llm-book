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
"""Generated from 06_01_qwen3.ipynb by tools/nb2book.py.

Edit the notebook, then run ``pixi run export``.
"""

from __future__ import annotations

from max.graph import Dim
from max.nn import ModuleList, RMSNorm
from max.tensor import Tensor

from minillm.attention import Attention
from minillm.config import LlamaConfig
from minillm.layers import apply_rope, rope_cos_sin
from minillm.model import DecoderLayer, LlamaForCausalLM, LlamaModel


class Qwen3Attention(Attention):
    """Llama attention with an RMSNorm on each query and key head."""

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__(config)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def project(
        self, x: Tensor, start: int | Dim | Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch, seq, _ = x.shape
        q = self.q_proj(x).reshape([batch, seq, self.num_heads, self.head_dim])
        k = self.k_proj(x).reshape(
            [batch, seq, self.num_kv_heads, self.head_dim]
        )
        v = self.v_proj(x).reshape(
            [batch, seq, self.num_kv_heads, self.head_dim]
        )
        q, k = self.q_norm(q), self.k_norm(k)
        cos, sin = rope_cos_sin(
            self.head_dim, self.rope_theta, start, seq, x.mesh
        )
        return apply_rope(q, cos, sin), apply_rope(k, cos, sin), v


class Qwen3DecoderLayer(DecoderLayer):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__(config)
        self.self_attn = Qwen3Attention(config)


class Qwen3Model(LlamaModel):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__(config)
        self.layers = ModuleList(
            [Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )


class Qwen3ForCausalLM(LlamaForCausalLM):
    """Qwen3: a Llama model whose attention normalizes queries and keys."""

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__(config)
        self.model = Qwen3Model(config)

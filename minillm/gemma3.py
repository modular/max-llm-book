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
"""Generated from 06_02_gemma3.ipynb by tools/nb2book.py.

Edit the notebook, then run ``pixi run export``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from max import functional as F
from max.driver import Device
from max.dtype import DType
from max.graph import Dim
from max.nn import Embedding, GemmaRMSNorm, Linear, Module, ModuleList
from max.nn.kernels.attention import flash_attention
from max.tensor import Tensor
from max.tensor.sharding import DeviceMesh

from minillm.attention import Attention, causal_mask
from minillm.config import LlamaConfig
from minillm.layers import MLP, apply_rope, rope_cos_sin


@dataclass(frozen=True)
class Gemma3Config(LlamaConfig):
    sliding_window: int = 512
    layer_types: tuple[str, ...] = ()
    rope_local_base_freq: float = 10000.0
    query_pre_attn_scalar: float = 256.0

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "layer_types", tuple(self.layer_types))


def sliding_window_mask(
    q_len: int | Dim,
    kv_len: int | Dim,
    window: int,
    device: Device | DeviceMesh,
) -> Tensor:
    """Like ``causal_mask``, but each query also can't see keys ``window`` or
    more positions behind it."""
    # A finite value, not -inf: a window hides a query's whole first block of
    # keys, and a blockwise attention kernel would turn that into NaN.
    blocked = F.broadcast_to(
        Tensor(-1e30, dtype=DType.float32, device=device), [q_len, kv_len]
    )
    return F.band_part(
        blocked, num_lower=window - 1, num_upper=kv_len - q_len, exclude=True
    )


class Gemma3Attention(Attention):
    def __init__(self, config: Gemma3Config, layer: int) -> None:
        super().__init__(config)
        self.sliding = config.layer_types[layer] == "sliding_attention"
        self.window = config.sliding_window
        self.rope_theta = (
            config.rope_local_base_freq if self.sliding else config.rope_theta
        )
        self.scale = config.query_pre_attn_scalar**-0.5
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(self, x: Tensor) -> Tensor:
        batch, seq, _ = x.shape
        q = self.q_norm(
            self.q_proj(x).reshape([batch, seq, self.num_heads, self.head_dim])
        )
        k = self.k_norm(
            self.k_proj(x).reshape(
                [batch, seq, self.num_kv_heads, self.head_dim]
            )
        )
        v = self.v_proj(x).reshape(
            [batch, seq, self.num_kv_heads, self.head_dim]
        )
        cos, sin = rope_cos_sin(self.head_dim, self.rope_theta, 0, seq, x.mesh)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if self.sliding:
            mask = sliding_window_mask(seq, seq, self.window, x.mesh)
        else:
            mask = causal_mask(seq, seq, x.mesh)
        mask = F.broadcast_to(mask, [batch, seq, seq])
        out = flash_attention(q, k, v, self.scale, mask).rebind_mapping(
            q.mapping
        )
        return self.o_proj(
            out.reshape([batch, seq, self.num_heads * self.head_dim])
        )


class GeGLU(MLP):
    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(
            F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x)
        )


class Gemma3DecoderLayer(Module):
    def __init__(self, config: Gemma3Config, layer: int) -> None:
        eps = config.rms_norm_eps
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=eps)
        self.self_attn = Gemma3Attention(config, layer)
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=eps
        )
        self.pre_feedforward_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=eps
        )
        self.mlp = GeGLU(config)
        self.post_feedforward_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=eps
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.post_attention_layernorm(
            self.self_attn(self.input_layernorm(x))
        )
        return x + self.post_feedforward_layernorm(
            self.mlp(self.pre_feedforward_layernorm(x))
        )


class Gemma3Model(Module):
    def __init__(self, config: Gemma3Config) -> None:
        self.embed_scale = math.sqrt(config.hidden_size)
        self.embed_tokens = Embedding(config.vocab_size, dim=config.hidden_size)
        self.layers = ModuleList(
            [
                Gemma3DecoderLayer(config, i)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: Tensor) -> Tensor:
        x = self.embed_tokens(input_ids) * self.embed_scale
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class Gemma3ForCausalLM(Module):
    def __init__(self, config: Gemma3Config) -> None:
        self.model = Gemma3Model(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids: Tensor) -> Tensor:
        return self.lm_head(self.model(input_ids))


def without_sliding_window(config: Gemma3Config) -> Gemma3Config:
    """Returns ``config`` with windows too wide to hide any token; each layer
    keeps its own RoPE base."""
    return replace(config, sliding_window=config.max_position_embeddings)

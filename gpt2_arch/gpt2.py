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
"""GPT-2 model implementation using max.experimental.nn.

This module defines the full GPT-2 model hierarchy:
  GPT2Config → GPT2MLP, GPT2MultiHeadAttention, LayerNorm
    → GPT2Block → MaxGPT2Model → MaxGPT2LMHeadModel

MaxGPT2LMHeadModel is imported by model.py and compiled by the serving
pipeline at startup.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import max.experimental.functional as F
from max.driver import Device
from max.dtype import DType
from max.experimental.nn import Embedding, Linear, Module, Sequential
from max.experimental.tensor import Tensor
from max.graph import Dim, DimLike, TensorValue


# ANCHOR: model_configuration
@dataclass
class GPT2Config:
    """GPT-2 configuration matching HuggingFace"""

    vocab_size: int = 50257
    n_positions: int = 1024
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    n_inner: int | None = None
    layer_norm_epsilon: float = 1e-5


# ANCHOR_END: model_configuration


# ANCHOR: feed_forward_network
class GPT2MLP(Module):  # type: ignore[type-arg]
    """Exact HuggingFace GPT-2 MLP structure"""

    def __init__(self, intermediate_size: int, config: GPT2Config) -> None:
        embed_dim = config.n_embd
        self.c_fc = Linear(embed_dim, intermediate_size, bias=True)
        self.c_proj = Linear(intermediate_size, embed_dim, bias=True)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = F.gelu(hidden_states, approximate="tanh")
        hidden_states = self.c_proj(hidden_states)
        return hidden_states


# ANCHOR_END: feed_forward_network


# ANCHOR: causal_mask
@F.functional
def causal_mask(
    sequence_length: DimLike,
    num_tokens: DimLike,
    *,
    dtype: DType,
    device: Device,
) -> Tensor:
    n = Dim(sequence_length) + num_tokens
    mask = Tensor(float("-inf"), dtype=dtype, device=device)
    mask = F.broadcast_to(mask, shape=(sequence_length, n))
    return F.band_part(mask, num_lower=None, num_upper=0, exclude=True)


# ANCHOR_END: causal_mask


# ANCHOR: multi_head_attention
class GPT2MultiHeadAttention(Module):  # type: ignore[type-arg]
    """Exact HuggingFace GPT-2 attention structure"""

    def __init__(self, config: GPT2Config) -> None:
        self.embed_dim = config.n_embd
        self.num_heads = config.n_head
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim

        self.c_attn = Linear(self.embed_dim, 3 * self.embed_dim, bias=True)
        self.c_proj = Linear(self.embed_dim, self.embed_dim, bias=True)

    def _attn(
        self,
        query: Tensor | TensorValue,
        key: Tensor | TensorValue,
        value: Tensor | TensorValue,
    ) -> Tensor | TensorValue:
        attn_weights = query @ key.transpose(-1, -2)

        # Scale attention weights
        attn_weights = attn_weights / math.sqrt(int(value.shape[-1]))

        # Apply causal mask
        seq_len = query.shape[-2]
        mask = causal_mask(seq_len, 0, dtype=query.dtype, device=query.device)
        attn_weights = attn_weights + mask

        attn_weights = F.softmax(attn_weights)
        attn_output = attn_weights @ value

        return attn_output

    def _split_heads(
        self, tensor: Tensor | TensorValue, num_heads: int, attn_head_size: int
    ) -> Tensor | TensorValue:
        """Split the last dimension into (num_heads, head_size)"""
        new_shape = list(tensor.shape[:-1]) + [num_heads, attn_head_size]
        tensor = tensor.reshape(new_shape)
        return tensor.transpose(
            -3, -2
        )  # (batch, head, seq_length, head_features)

    def _merge_heads(
        self, tensor: Tensor | TensorValue, num_heads: int, attn_head_size: int
    ) -> Tensor | TensorValue:
        """Merge attention heads back"""
        tensor = tensor.transpose(-3, -2)
        new_shape = list(tensor.shape[:-2]) + [num_heads * attn_head_size]
        return tensor.reshape(new_shape)

    def forward(self, hidden_states: Tensor) -> Tensor:
        split_result = F.split(
            self.c_attn(hidden_states),
            [self.split_size, self.split_size, self.split_size],
            axis=2,
        )
        query = cast(Tensor | TensorValue, split_result[0])
        key = cast(Tensor | TensorValue, split_result[1])
        value = cast(Tensor | TensorValue, split_result[2])

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        attn_output = self._attn(query, key, value)
        attn_output = self._merge_heads(
            attn_output, self.num_heads, self.head_dim
        )
        attn_output = self.c_proj(cast(Tensor, attn_output))

        return cast(Tensor, attn_output)


# ANCHOR_END: multi_head_attention


# ANCHOR: layer_normalization
class LayerNorm(Module):  # type: ignore[type-arg]
    def __init__(self, dim: DimLike, *, eps: float = 1e-5) -> None:
        self.eps = eps
        self.weight = Tensor.ones([dim])
        self.bias = Tensor.zeros([dim])

    def forward(self, x: Tensor) -> Tensor:
        return F.layer_norm(
            x, gamma=self.weight, beta=self.bias, epsilon=self.eps
        )


# ANCHOR_END: layer_normalization


# ANCHOR: transformer_block
class GPT2Block(Module):  # type: ignore[type-arg]
    """Exact HuggingFace GPT-2 transformer block structure"""

    def __init__(self, config: GPT2Config) -> None:
        hidden_size = config.n_embd
        inner_dim = (
            config.n_inner if config.n_inner is not None else 4 * hidden_size
        )

        self.ln_1 = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPT2MultiHeadAttention(config)
        self.ln_2 = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(inner_dim, config)

    def forward(self, hidden_states: Tensor) -> Tensor:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output = self.attn(hidden_states)
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = residual + feed_forward_hidden_states

        return hidden_states


# ANCHOR_END: transformer_block


# ANCHOR: stacking_transformer_blocks
class MaxGPT2Model(Module):  # type: ignore[type-arg]
    def __init__(
        self,
        config: GPT2Config,
    ) -> None:
        self.wte = Embedding(config.vocab_size, dim=config.n_embd)
        self.wpe = Embedding(config.n_positions, dim=config.n_embd)
        self.h = Sequential(*(GPT2Block(config) for _ in range(config.n_layer)))
        self.ln_f = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

    def forward(self, input_ids: Tensor) -> Tensor:
        _, seq_length = input_ids.shape
        tok_embeds = self.wte(input_ids)
        pos_embeds = self.wpe(
            Tensor.arange(
                seq_length, dtype=input_ids.dtype, device=input_ids.device
            )
        )
        x = tok_embeds + pos_embeds
        x = self.h(x)
        x = self.ln_f(x)
        return x


# ANCHOR_END: stacking_transformer_blocks


# ANCHOR: language_model_head
class MaxGPT2LMHeadModel(Module):  # type: ignore[type-arg]
    """Exact HuggingFace GPT-2 model structure"""

    def __init__(self, config: GPT2Config) -> None:
        self.config = config
        self.transformer = MaxGPT2Model(config)
        self.lm_head = Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, input_ids: Tensor) -> Tensor:
        input_ids = self.transformer(input_ids)
        return self.lm_head(input_ids)


# ANCHOR_END: language_model_head

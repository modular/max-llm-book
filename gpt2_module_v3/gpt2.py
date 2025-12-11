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

"""Implements the GPT-2 model for MAX serving."""

from __future__ import annotations

import math
from collections.abc import Sequence

from max.dtype import DType
from max.driver import Device
from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.graph import BufferValue, Dim, DimLike, TensorValue
from max.kv_cache import NullKVCacheManager, PagedKVCacheManager
from max.nn.kv_cache import PagedCacheValues
from max.nn.module_v3 import Module
from max.nn.module_v3.embedding import Embedding
from max.nn.module_v3.linear import Linear
from max.nn.module_v3.sequential import Sequential

from .model_config import GPT2Config


@F.functional
def causal_mask(
    sequence_length: DimLike,
    num_tokens: DimLike,
    *,
    dtype: DType,
    device: Device,
):
    """Create a causal attention mask."""
    n = Dim(sequence_length) + num_tokens
    mask = Tensor.constant(float("-inf"), dtype=dtype, device=device)
    mask = F.broadcast_to(mask, shape=(sequence_length, n))
    return F.band_part(mask, num_lower=None, num_upper=0, exclude=True)


class LayerNorm(Module):
    """Layer normalization module."""

    def __init__(self, dim: DimLike, *, eps: float = 1e-5):
        self.eps = eps
        self.weight = Tensor.ones([dim])
        self.bias = Tensor.zeros([dim])

    def __call__(self, x: Tensor) -> Tensor:
        return F.layer_norm(x, gamma=self.weight, beta=self.bias, epsilon=self.eps)


class GPT2Attention(Module):
    """GPT-2 attention matching HuggingFace structure."""

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.embed_dim = config.n_embd
        self.num_heads = config.n_head
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim

        self.c_attn = Linear(self.embed_dim, 3 * self.embed_dim, bias=True)
        self.c_proj = Linear(self.embed_dim, self.embed_dim, bias=True)

    def _attn(self, query, key, value):
        attn_weights = query @ key.transpose(-1, -2)
        attn_weights = attn_weights / math.sqrt(int(value.shape[-1]))

        seq_len = query.shape[-2]
        mask = causal_mask(seq_len, 0, dtype=query.dtype, device=query.device)
        attn_weights = attn_weights + mask

        attn_weights = F.softmax(attn_weights)
        attn_output = attn_weights @ value

        return attn_output

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """Split the last dimension into (num_heads, head_size)."""
        new_shape = tensor.shape[:-1] + [num_heads, attn_head_size]
        tensor = tensor.reshape(new_shape)
        return tensor.transpose(-3, -2)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """Merge attention heads back."""
        tensor = tensor.transpose(-3, -2)
        new_shape = tensor.shape[:-2] + [num_heads * attn_head_size]
        return tensor.reshape(new_shape)

    def __call__(self, hidden_states):
        query, key, value = F.split(
            self.c_attn(hidden_states),
            [self.split_size, self.split_size, self.split_size],
            axis=2,
        )

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        attn_output = self._attn(query, key, value)
        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)

        return attn_output


class GPT2MLP(Module):
    """GPT-2 MLP structure."""

    def __init__(self, intermediate_size: int, config: GPT2Config):
        super().__init__()
        embed_dim = config.n_embd
        self.c_fc = Linear(embed_dim, intermediate_size, bias=True)
        self.c_proj = Linear(intermediate_size, embed_dim, bias=True)

    def __call__(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = F.gelu(hidden_states, approximate="tanh")
        hidden_states = self.c_proj(hidden_states)
        return hidden_states


class GPT2Block(Module):
    """GPT-2 transformer block."""

    def __init__(self, config: GPT2Config):
        super().__init__()
        hidden_size = config.n_embd
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.ln_1 = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config)
        self.ln_2 = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(inner_dim, config)

    def __call__(self, hidden_states):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output = self.attn(hidden_states)
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = residual + feed_forward_hidden_states

        return hidden_states


class GPT2TextModel(Module):
    """The GPT-2 language model."""

    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.devices = config.devices

        self.wte = Embedding(config.vocab_size, dim=config.n_embd)
        self.wpe = Embedding(config.n_positions, dim=config.n_embd)
        self.h = Sequential(*(GPT2Block(config) for _ in range(config.n_layer)))
        self.ln_f = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.lm_head = Linear(config.n_embd, config.vocab_size, bias=False)

        self.n_embd = config.n_embd
        self.kv_params = config.kv_params
        self.return_logits = config.return_logits

    def __call__(
        self,
        tokens: Tensor,
        kv_collection: PagedCacheValues,
        return_n_logits: Tensor,
        input_row_offsets: Tensor,
    ) -> tuple[Tensor, ...]:
        # Get sequence length from tokens
        seq_length = tokens.shape[0]

        # Token embeddings
        tok_embeds = self.wte(tokens)

        # Position embeddings using Tensor.arange like the original implementation
        positions = Tensor.arange(
            seq_length, dtype=tokens.dtype, device=tokens.device
        )
        pos_embeds = self.wpe(positions)

        # Combine embeddings
        h = tok_embeds + pos_embeds

        # Add batch dimension for transformer layers (they expect batch dim)
        h = h.reshape([1, seq_length, self.n_embd])

        # Run through transformer layers
        h = self.h(h)
        h = self.ln_f(h)

        # Remove batch dimension
        h = h.reshape([seq_length, self.n_embd])

        # Get last token logits per sequence using input_row_offsets
        last_token_indices = input_row_offsets[1:] - 1
        last_token_h = F.gather(h, last_token_indices, axis=0)
        last_logits = F.cast(self.lm_head(last_token_h), DType.float32)

        return (last_logits,)


class GPT2(Module):
    """The GPT-2 model wrapper for serving."""

    def __init__(
        self,
        config: GPT2Config,
        kv_manager: PagedKVCacheManager | NullKVCacheManager,
    ) -> None:
        super().__init__()
        self.language_model = GPT2TextModel(config)
        self.config = config
        self.kv_manager = kv_manager

    def __call__(
        self,
        tokens: Tensor,
        return_n_logits: Tensor,
        input_row_offsets: Tensor,
        *variadic_args,
    ) -> tuple[Tensor, ...]:
        kv_collection = _unflatten_kv_inputs(
            self.config, self.kv_manager, variadic_args
        )
        return self.language_model(
            tokens, kv_collection[0], return_n_logits, input_row_offsets
        )


def _unflatten_kv_inputs(
    config: GPT2Config,
    kv_manager: PagedKVCacheManager | NullKVCacheManager,
    kv_inputs_flat: Sequence[Tensor],
) -> list[PagedCacheValues]:
    """Unflatten KV cache inputs from variadic args."""
    kv_params = config.kv_params
    n_devices = kv_params.n_devices
    fetch_types = kv_manager.get_symbolic_inputs()[0]
    len_of_kv_tuple_per_dev = len(list(fetch_types))
    kv_caches_per_dev: list[PagedCacheValues] = []

    for i in range(n_devices):
        start_idx = i * len_of_kv_tuple_per_dev

        kv_block = kv_inputs_flat[start_idx]
        cache_lengths = kv_inputs_flat[start_idx + 1]
        lookup_table = kv_inputs_flat[start_idx + 2]
        max_lengths = kv_inputs_flat[start_idx + 3]

        kv_caches_per_dev.append(
            PagedCacheValues(
                kv_blocks=BufferValue(kv_block),
                cache_lengths=TensorValue(cache_lengths),
                lookup_table=TensorValue(lookup_table),
                max_lengths=TensorValue(max_lengths),
            )
        )
    return kv_caches_per_dev

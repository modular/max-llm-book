"""
Solution for Step 04: Multi-head Attention

This module implements multi-head attention, which allows the model to jointly
attend to information from different representation subspaces at different positions.
"""

import math

from max.driver import Device
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.graph import Dim, DimLike
from max.nn.module_v3 import Linear, Module

from solution_01 import GPT2Config
from solution_03 import causal_mask


class GPT2MultiHeadAttention(Module):
    """Multi-head attention for GPT-2, matching HuggingFace structure."""

    def __init__(self, config: GPT2Config):
        """Initialize multi-head attention.

        Args:
            config: GPT2Config containing n_embd and n_head
        """
        super().__init__()

        self.embed_dim = config.n_embd
        self.num_heads = config.n_head
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim

        # Combined Q/K/V projection
        self.c_attn = Linear(self.embed_dim, 3 * self.embed_dim, bias=True)
        # Output projection
        self.c_proj = Linear(self.embed_dim, self.embed_dim, bias=True)

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """Split the last dimension into (num_heads, head_size).

        Transforms shape from [batch, seq_length, n_embd]
        to [batch, num_heads, seq_length, head_size]

        Args:
            tensor: Input tensor, shape [batch, seq_length, n_embd]
            num_heads: Number of attention heads
            attn_head_size: Dimension of each head

        Returns:
            Tensor with shape [batch, num_heads, seq_length, head_size]
        """
        # Add head dimension: [batch, seq_length, n_embd] -> [batch, seq_length, num_heads, head_size]
        new_shape = tensor.shape[:-1] + [num_heads, attn_head_size]
        tensor = tensor.reshape(new_shape)
        # Move heads dimension: [batch, seq_length, num_heads, head_size] -> [batch, num_heads, seq_length, head_size]
        return tensor.transpose(-3, -2)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """Merge attention heads back to original shape.

        Transforms shape from [batch, num_heads, seq_length, head_size]
        to [batch, seq_length, n_embd]

        Args:
            tensor: Input tensor, shape [batch, num_heads, seq_length, head_size]
            num_heads: Number of attention heads
            attn_head_size: Dimension of each head

        Returns:
            Tensor with shape [batch, seq_length, n_embd]
        """
        # Move heads dimension back: [batch, num_heads, seq_length, head_size] -> [batch, seq_length, num_heads, head_size]
        tensor = tensor.transpose(-3, -2)
        # Flatten head dimensions: [batch, seq_length, num_heads, head_size] -> [batch, seq_length, n_embd]
        new_shape = tensor.shape[:-2] + [num_heads * attn_head_size]
        return tensor.reshape(new_shape)

    def _attn(self, query, key, value):
        """Compute attention for all heads in parallel.

        Args:
            query: Query tensor, shape [batch, num_heads, seq_length, head_size]
            key: Key tensor, shape [batch, num_heads, seq_length, head_size]
            value: Value tensor, shape [batch, num_heads, seq_length, head_size]

        Returns:
            Attention output, shape [batch, num_heads, seq_length, head_size]
        """
        # Compute attention scores
        attn_weights = query @ key.transpose(-1, -2)

        # Scale attention weights
        attn_weights = attn_weights / math.sqrt(int(value.shape[-1]))

        # Apply causal mask
        seq_len = query.shape[-2]
        mask = causal_mask(seq_len, 0, dtype=query.dtype, device=query.device)
        attn_weights = attn_weights + mask

        # Softmax and weighted sum
        attn_weights = F.softmax(attn_weights)
        attn_output = attn_weights @ value

        return attn_output

    def __call__(self, hidden_states):
        """Apply multi-head attention.

        Args:
            hidden_states: Input tensor, shape [batch, seq_length, n_embd]

        Returns:
            Attention output, shape [batch, seq_length, n_embd]
        """
        # Project to Q, K, V
        qkv = self.c_attn(hidden_states)
        query, key, value = F.split(
            qkv, [self.split_size, self.split_size, self.split_size], axis=-1
        )

        # Split into multiple heads
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        # Apply attention
        attn_output = self._attn(query, key, value)

        # Merge heads back
        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)

        # Output projection
        attn_output = self.c_proj(attn_output)

        return attn_output

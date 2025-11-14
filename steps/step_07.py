"""
Step 07: Multi-head Attention

Implement multi-head attention that splits Q/K/V into multiple heads,
computes attention in parallel for each head, and merges the results.

Tasks:
1. Import required modules (math, F, Tensor, Linear, Module, etc.)
2. Create c_attn and c_proj linear layers
3. Implement _split_heads: reshape and transpose to add head dimension
4. Implement _merge_heads: transpose and reshape to remove head dimension
5. Implement _attn: compute attention for all heads in parallel
6. Implement forward pass: project -> split -> attend -> merge -> project

Run: pixi run s07
"""

# TODO: Import required modules
# Hint: You'll need math for scaling
# Hint: You'll need functional as F from max.experimental
# Hint: You'll need Tensor, Device, DType from max.experimental.tensor and max.driver
# Hint: You'll need Dim, DimLike from max.graph
# Hint: You'll also need Linear and Module from max.nn.module_v3

from solutions.solution_01 import GPT2Config


# TODO: Copy causal_mask function from solution_02.py
# This is the same function you implemented in Step 02


class GPT2MultiHeadAttention(Module):
    """Multi-head attention for GPT-2."""

    def __init__(self, config: GPT2Config):
        super().__init__()

        self.embed_dim = config.n_embd
        self.num_heads = config.n_head
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim

        # TODO: Create combined Q/K/V projection
        # Hint: Use Linear(self.embed_dim, 3 * self.embed_dim, bias=True)
        self.c_attn = None

        # TODO: Create output projection
        # Hint: Use Linear(self.embed_dim, self.embed_dim, bias=True)
        self.c_proj = None

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """Split the last dimension into (num_heads, head_size).

        Args:
            tensor: Input tensor, shape [batch, seq_length, n_embd]
            num_heads: Number of attention heads
            attn_head_size: Dimension of each head

        Returns:
            Tensor with shape [batch, num_heads, seq_length, head_size]
        """
        # TODO: Add head dimension
        # Hint: new_shape = tensor.shape[:-1] + [num_heads, attn_head_size]
        # Hint: tensor = tensor.reshape(new_shape)
        pass

        # TODO: Move heads dimension to position 1
        # Hint: return tensor.transpose(-3, -2)
        return None

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """Merge attention heads back to original shape.

        Args:
            tensor: Input tensor, shape [batch, num_heads, seq_length, head_size]
            num_heads: Number of attention heads
            attn_head_size: Dimension of each head

        Returns:
            Tensor with shape [batch, seq_length, n_embd]
        """
        # TODO: Move heads dimension back
        # Hint: tensor = tensor.transpose(-3, -2)
        pass

        # TODO: Flatten head dimensions
        # Hint: new_shape = tensor.shape[:-2] + [num_heads * attn_head_size]
        # Hint: return tensor.reshape(new_shape)
        return None

    def _attn(self, query, key, value):
        """Compute attention for all heads in parallel.

        Args:
            query: Query tensor, shape [batch, num_heads, seq_length, head_size]
            key: Key tensor, shape [batch, num_heads, seq_length, head_size]
            value: Value tensor, shape [batch, num_heads, seq_length, head_size]

        Returns:
            Attention output, shape [batch, num_heads, seq_length, head_size]
        """
        # TODO: Implement attention computation
        # The same 5-step process: scores, scale, mask, softmax, weighted sum
        # Hint: Compute attention scores: query @ key.transpose(-1, -2)
        # Hint: Scale by sqrt(head_dim): attn_weights / math.sqrt(head_dim)
        # Hint: Apply causal mask using causal_mask function
        # Hint: Apply softmax: F.softmax(attn_weights)
        # Hint: Weighted sum: attn_weights @ value
        return None

    def __call__(self, hidden_states):
        """Apply multi-head attention.

        Args:
            hidden_states: Input tensor, shape [batch, seq_length, n_embd]

        Returns:
            Attention output, shape [batch, seq_length, n_embd]
        """
        # TODO: Project to Q, K, V
        # Hint: qkv = self.c_attn(hidden_states)
        # Hint: query, key, value = F.split(qkv, [self.split_size, self.split_size, self.split_size], axis=-1)
        pass

        # TODO: Split into multiple heads
        # Hint: query = self._split_heads(query, self.num_heads, self.head_dim)
        # Hint: key = self._split_heads(key, self.num_heads, self.head_dim)
        # Hint: value = self._split_heads(value, self.num_heads, self.head_dim)
        pass

        # TODO: Apply attention
        # Hint: attn_output = self._attn(query, key, value)
        pass

        # TODO: Merge heads back
        # Hint: attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        pass

        # TODO: Output projection
        # Hint: attn_output = self.c_proj(attn_output)
        # Hint: return attn_output
        return None

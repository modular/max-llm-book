"""
Solution for Step 06: Transformer Block

This module implements a complete GPT-2 transformer block, combining
multi-head attention, MLP, layer normalization, and residual connections.
"""

from max.nn.module_v3 import Module

from step_01 import GPT2Config
from step_02 import GPT2MLP
from step_04 import GPT2MultiHeadAttention
from step_05 import LayerNorm


class GPT2Block(Module):
    """Complete GPT-2 transformer block matching HuggingFace structure.

    Architecture (pre-norm):
    1. x = x + attention(layer_norm(x))
    2. x = x + mlp(layer_norm(x))
    """

    def __init__(self, config: GPT2Config):
        """Initialize transformer block.

        Args:
            config: GPT2Config containing model hyperparameters
        """
        super().__init__()

        hidden_size = config.n_embd
        # Inner dimension for MLP (4x hidden size by default)
        inner_dim = (
            config.n_inner
            if hasattr(config, "n_inner") and config.n_inner is not None
            else 4 * hidden_size
        )

        # First layer norm (before attention)
        self.ln_1 = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        # Multi-head attention
        self.attn = GPT2MultiHeadAttention(config)
        # Second layer norm (before MLP)
        self.ln_2 = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        # Feed-forward MLP
        self.mlp = GPT2MLP(inner_dim, config)

    def __call__(self, hidden_states):
        """Apply transformer block.

        Args:
            hidden_states: Input tensor, shape [batch, seq_length, n_embd]

        Returns:
            Output tensor, shape [batch, seq_length, n_embd]
        """
        # Attention block with residual connection
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output = self.attn(hidden_states)
        hidden_states = attn_output + residual

        # MLP block with residual connection
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = residual + feed_forward_hidden_states

        return hidden_states

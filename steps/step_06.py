"""
Step 06: Transformer Block

Combine multi-head attention, MLP, layer normalization, and residual
connections into a complete transformer block.

Tasks:
1. Import Module and all previous solution components
2. Create ln_1, attn, ln_2, and mlp layers
3. Implement forward pass with pre-norm residual pattern

Run: pixi run s06
"""

# TODO: Import required modules
# Hint: You'll need Module from max.nn.module_v3

from step_01 import GPT2Config
from step_02 import GPT2MLP
from step_04 import GPT2MultiHeadAttention
from step_05 import LayerNorm


class GPT2Block(Module):
    """Complete GPT-2 transformer block."""

    def __init__(self, config: GPT2Config):
        """Initialize transformer block.

        Args:
            config: GPT2Config containing model hyperparameters
        """
        super().__init__()

        hidden_size = config.n_embd
        inner_dim = (
            config.n_inner
            if hasattr(config, "n_inner") and config.n_inner is not None
            else 4 * hidden_size
        )

        # TODO: Create first layer norm (before attention)
        # Hint: Use LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.ln_1 = None

        # TODO: Create multi-head attention
        # Hint: Use GPT2MultiHeadAttention(config)
        self.attn = None

        # TODO: Create second layer norm (before MLP)
        # Hint: Use LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.ln_2 = None

        # TODO: Create MLP
        # Hint: Use GPT2MLP(inner_dim, config)
        self.mlp = None

    def __call__(self, hidden_states):
        """Apply transformer block.

        Args:
            hidden_states: Input tensor, shape [batch, seq_length, n_embd]

        Returns:
            Output tensor, shape [batch, seq_length, n_embd]
        """
        # TODO: Attention block with residual connection
        # Hint: residual = hidden_states
        # Hint: hidden_states = self.ln_1(hidden_states)
        # Hint: attn_output = self.attn(hidden_states)
        # Hint: hidden_states = attn_output + residual
        pass

        # TODO: MLP block with residual connection
        # Hint: residual = hidden_states
        # Hint: hidden_states = self.ln_2(hidden_states)
        # Hint: feed_forward_hidden_states = self.mlp(hidden_states)
        # Hint: hidden_states = residual + feed_forward_hidden_states
        pass

        # TODO: Return the output
        return None

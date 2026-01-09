"""
Solution for Step 02: Feed-forward Network (MLP)

This module implements the feed-forward network (MLP) used in each
transformer block with GELU activation.
"""

from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.nn.module_v3 import Linear, Module

from solution_01 import GPT2Config


class GPT2MLP(Module):
    """Feed-forward network matching HuggingFace GPT-2 structure.

    Args:
        intermediate_size: Size of the intermediate layer.
        config: GPT-2 configuration.
    """

    def __init__(self, intermediate_size: int, config: GPT2Config):
        super().__init__()
        embed_dim = config.n_embd
        self.c_fc = Linear(embed_dim, intermediate_size, bias=True)
        self.c_proj = Linear(intermediate_size, embed_dim, bias=True)

    def __call__(self, hidden_states: Tensor) -> Tensor:
        """Apply feed-forward network.

        Args:
            hidden_states: Input hidden states.

        Returns:
            MLP output.
        """
        hidden_states = self.c_fc(hidden_states)
        hidden_states = F.gelu(hidden_states, approximate="tanh")
        hidden_states = self.c_proj(hidden_states)
        return hidden_states

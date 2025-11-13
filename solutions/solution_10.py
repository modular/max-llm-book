"""
Solution for Step 10: Stacking Transformer Blocks

This module stacks multiple transformer blocks and adds embeddings
to create the complete GPT-2 transformer architecture.
"""

from max.experimental.tensor import Tensor
from max.nn.module_v3 import Embedding, Module, Sequential

from solutions.solution_01 import GPT2Config
from solutions.solution_08 import LayerNorm
from solutions.solution_09 import GPT2Block


class GPT2Model(Module):
    """Complete GPT-2 transformer model matching HuggingFace structure.

    Architecture:
    1. Token embeddings + position embeddings
    2. Stack of n_layer transformer blocks
    3. Final layer normalization
    """

    def __init__(self, config: GPT2Config):
        """Initialize GPT-2 model.

        Args:
            config: GPT2Config containing model hyperparameters
        """
        super().__init__()

        # Token embeddings (vocabulary -> embeddings)
        self.wte = Embedding(config.vocab_size, dim=config.n_embd)
        # Position embeddings (positions -> embeddings)
        self.wpe = Embedding(config.n_positions, dim=config.n_embd)

        # Stack of transformer blocks
        self.h = Sequential(*(GPT2Block(config) for _ in range(config.n_layer)))

        # Final layer normalization
        self.ln_f = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

    def __call__(self, input_ids):
        """Forward pass through the transformer.

        Args:
            input_ids: Token IDs, shape [batch, seq_length]

        Returns:
            Hidden states, shape [batch, seq_length, n_embd]
        """
        batch_size, seq_length = input_ids.shape

        # Get token embeddings
        tok_embeds = self.wte(input_ids)

        # Get position embeddings
        pos_embeds = self.wpe(
            Tensor.arange(seq_length, dtype=input_ids.dtype, device=input_ids.device)
        )

        # Combine embeddings
        x = tok_embeds + pos_embeds

        # Apply transformer blocks
        x = self.h(x)

        # Final layer norm
        x = self.ln_f(x)

        return x

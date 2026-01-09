"""
Step 07: Stacking Transformer Blocks

Stack multiple transformer blocks with embeddings to create
the complete GPT-2 model architecture.

Tasks:
1. Import Tensor, Embedding, Module, Sequential, and previous components
2. Create token and position embeddings
3. Stack n_layer transformer blocks using Sequential
4. Create final layer normalization
5. Implement forward pass: embeddings -> blocks -> layer norm

Run: pixi run s10
"""

# TODO: Import required modules
# Hint: You'll need Tensor from max.experimental.tensor
# Hint: You'll need Embedding, Module, Sequential from max.nn.module_v3

from step_01 import GPT2Config
from step_05 import LayerNorm
from step_06 import GPT2Block


class MaxGPT2Model(Module):
    """Complete GPT-2 transformer model."""

    def __init__(self, config: GPT2Config):
        """Initialize GPT-2 model.

        Args:
            config: GPT2Config containing model hyperparameters
        """
        super().__init__()

        # TODO: Create token embeddings
        # Hint: Use Embedding(config.vocab_size, dim=config.n_embd)
        self.wte = None

        # TODO: Create position embeddings
        # Hint: Use Embedding(config.n_positions, dim=config.n_embd)
        self.wpe = None

        # TODO: Stack transformer blocks
        # Hint: Use Sequential(*(GPT2Block(config) for _ in range(config.n_layer)))
        # This creates config.n_layer blocks (12 for GPT-2 base)
        self.h = None

        # TODO: Create final layer normalization
        # Hint: Use LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.ln_f = None

    def __call__(self, input_ids):
        """Forward pass through the transformer.

        Args:
            input_ids: Token IDs, shape [batch, seq_length]

        Returns:
            Hidden states, shape [batch, seq_length, n_embd]
        """
        # TODO: Get batch size and sequence length
        # Hint: batch_size, seq_length = input_ids.shape
        pass

        # TODO: Get token embeddings
        # Hint: tok_embeds = self.wte(input_ids)
        pass

        # TODO: Get position embeddings
        # Hint: Create position indices with Tensor.arange(seq_length, dtype=input_ids.dtype, device=input_ids.device)
        # Hint: pos_embeds = self.wpe(position_indices)
        pass

        # TODO: Combine embeddings
        # Hint: x = tok_embeds + pos_embeds
        pass

        # TODO: Apply transformer blocks
        # Hint: x = self.h(x)
        pass

        # TODO: Apply final layer norm
        # Hint: x = self.ln_f(x)
        pass

        # TODO: Return the output
        return None

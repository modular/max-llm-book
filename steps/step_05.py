"""
Step 05: Token Embeddings

Implement token embeddings that convert discrete token IDs into continuous vectors.

Tasks:
1. Import Embedding and Module from max.nn.module_v3
2. Create token embedding layer using Embedding(vocab_size, dim=n_embd)
3. Implement forward pass that looks up embeddings for input token IDs

Run: pixi run s05
"""

# TODO: Import required modules from MAX
# Hint: You'll need Embedding and Module from max.nn.module_v3

from solutions.solution_01 import GPT2Config


class GPT2Embeddings(Module):
    """Token embeddings for GPT-2."""

    def __init__(self, config: GPT2Config):
        super().__init__()

        # TODO: Create token embedding layer
        # Hint: Use Embedding(config.vocab_size, dim=config.n_embd)
        # This creates a lookup table that converts token IDs to embedding vectors
        self.wte = None

    def __call__(self, input_ids):
        """Convert token IDs to embeddings.

        Args:
            input_ids: Tensor of token IDs, shape [batch_size, seq_length]

        Returns:
            Token embeddings, shape [batch_size, seq_length, n_embd]
        """
        # TODO: Return the embedded tokens
        # Hint: Simply call self.wte with input_ids
        return None

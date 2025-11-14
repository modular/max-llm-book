"""
Step 06: Position Embeddings

Implement position embeddings that encode sequence order information.

Tasks:
1. Import Tensor from max.experimental.tensor
2. Import Embedding and Module from max.nn.module_v3
3. Create position embedding layer using Embedding(n_positions, dim=n_embd)
4. Implement forward pass that looks up embeddings for position indices

Run: pixi run s06
"""

# TODO: Import required modules from MAX
# Hint: You'll need Tensor from max.experimental.tensor
# Hint: You'll need Embedding and Module from max.nn.module_v3

from solutions.solution_01 import GPT2Config


class GPT2PositionEmbeddings(Module):
    """Position embeddings for GPT-2."""

    def __init__(self, config: GPT2Config):
        super().__init__()

        # TODO: Create position embedding layer
        # Hint: Use Embedding(config.n_positions, dim=config.n_embd)
        # This creates a lookup table for position indices (0, 1, 2, ..., n_positions-1)
        self.wpe = None

    def __call__(self, position_ids):
        """Convert position indices to embeddings.

        Args:
            position_ids: Tensor of position indices, shape [seq_length] or [batch_size, seq_length]

        Returns:
            Position embeddings, shape matching input with added embedding dimension
        """
        # TODO: Return the position embeddings
        # Hint: Simply call self.wpe with position_ids
        return None

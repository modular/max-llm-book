"""
Solution for Step 06: Position Embeddings

This module implements position embeddings that encode sequence order information
into the transformer model.
"""

from max.experimental.tensor import Tensor
from max.nn.module_v3 import Embedding, Module

from solutions.solution_01 import GPT2Config


class GPT2PositionEmbeddings(Module):
    """Position embeddings for GPT-2, matching HuggingFace structure."""

    def __init__(self, config: GPT2Config):
        """Initialize position embedding layer.

        Args:
            config: GPT2Config containing n_positions and n_embd
        """
        super().__init__()

        # Position embedding: lookup table from position indices to embedding vectors
        # This encodes "where" information - position 0, 1, 2, etc.
        self.wpe = Embedding(config.n_positions, dim=config.n_embd)

    def __call__(self, position_ids):
        """Convert position indices to embeddings.

        Args:
            position_ids: Tensor of position indices, shape [seq_length] or [batch_size, seq_length]

        Returns:
            Position embeddings, shape matching input with added embedding dimension
        """
        # Simple lookup: each position index becomes its corresponding embedding vector
        return self.wpe(position_ids)

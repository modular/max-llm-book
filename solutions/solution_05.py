"""
Solution for Step 05: Token Embeddings

This module implements token embeddings that convert discrete token IDs
into continuous vector representations.
"""

from max.nn.module_v3 import Embedding, Module

from solutions.solution_01 import GPT2Config


class GPT2Embeddings(Module):
    """Token embeddings for GPT-2, matching HuggingFace structure."""

    def __init__(self, config: GPT2Config):
        """Initialize token embedding layer.

        Args:
            config: GPT2Config containing vocab_size and n_embd
        """
        super().__init__()

        # Token embedding: lookup table from vocab_size to embedding dimension
        # This converts discrete token IDs (0 to vocab_size-1) into dense vectors
        self.wte = Embedding(config.vocab_size, dim=config.n_embd)

    def __call__(self, input_ids):
        """Convert token IDs to embeddings.

        Args:
            input_ids: Tensor of token IDs, shape [batch_size, seq_length]

        Returns:
            Token embeddings, shape [batch_size, seq_length, n_embd]
        """
        # Simple lookup: each token ID becomes its corresponding embedding vector
        return self.wte(input_ids)

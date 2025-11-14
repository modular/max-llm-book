"""
Solution for Step 01: Model Configuration

This module implements the GPT-2 configuration dataclass that stores
hyperparameters matching HuggingFace's GPT-2 model structure.
"""

from dataclasses import dataclass


@dataclass
class GPT2Config:
    """GPT-2 configuration matching HuggingFace.

    Attributes:
        vocab_size: Size of the vocabulary.
        n_positions: Maximum sequence length.
        n_embd: Embedding dimension.
        n_layer: Number of transformer layers.
        n_head: Number of attention heads.
        n_inner: Inner dimension of feed-forward network (defaults to 4 * n_embd if None).
        layer_norm_epsilon: Epsilon for layer normalization.
    """

    vocab_size: int = 50257
    n_positions: int = 1024
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    n_inner: int = 3072
    layer_norm_epsilon: float = 1e-5

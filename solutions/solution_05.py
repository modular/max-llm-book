"""
Solution for Step 05: Layer Normalization

This module implements layer normalization that normalizes activations
across the embedding dimension for training stability.
"""

from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.graph import DimLike
from max.nn.module_v3 import Module


class LayerNorm(Module):
    """Layer normalization module.

    Args:
        dim: Dimension to normalize over.
        eps: Epsilon for numerical stability.
    """

    def __init__(self, dim: DimLike, *, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = Tensor.ones([dim])
        self.bias = Tensor.zeros([dim])

    def __call__(self, x: Tensor) -> Tensor:
        """Apply layer normalization.

        Args:
            x: Input tensor.

        Returns:
            Normalized tensor.
        """
        return F.layer_norm(x, gamma=self.weight, beta=self.bias, epsilon=self.eps)

"""
Step 05: Layer Normalization

Implement layer normalization that normalizes activations for training stability.

Tasks:
1. Import functional module (as F) and Tensor from max.experimental
2. Initialize learnable weight (gamma) and bias (beta) parameters
3. Apply layer normalization using F.layer_norm in the forward pass

Run: pixi run s05
"""

# 1: Import the required modules from MAX
# TODO: Import functional module from max.experimental with the alias F
# https://docs.modular.com/max/api/python/experimental/functional

# TODO: Import Tensor from max.experimental.tensor
# https://docs.modular.com/max/api/python/experimental/tensor.Tensor

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

        # 2: Initialize learnable weight and bias parameters
        # TODO: Create self.weight as a Tensor of ones with shape [dim]
        # https://docs.modular.com/max/api/python/experimental/tensor#max.experimental.tensor.Tensor.ones
        # Hint: This is the gamma parameter in layer normalization
        self.weight = None

        # TODO: Create self.bias as a Tensor of zeros with shape [dim]
        # https://docs.modular.com/max/api/python/experimental/tensor#max.experimental.tensor.Tensor.zeros
        # Hint: This is the beta parameter in layer normalization
        self.bias = None

    def __call__(self, x: Tensor) -> Tensor:
        """Apply layer normalization.

        Args:
            x: Input tensor.

        Returns:
            Normalized tensor.
        """
        # 3: Apply layer normalization and return the result
        # TODO: Use F.layer_norm() with x, gamma=self.weight, beta=self.bias, epsilon=self.eps
        # https://docs.modular.com/max/api/python/experimental/functional#max.experimental.functional.layer_norm
        # Hint: Layer normalization normalizes across the last dimension
        return None

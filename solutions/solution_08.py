"""
Solution for Step 08: Residual Connections and Layer Normalization

This module implements layer normalization and demonstrates residual connections,
which are essential for training deep transformer networks.
"""

from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.graph import DimLike
from max.nn.module_v3 import Module


class LayerNorm(Module):
    """Layer normalization module matching HuggingFace GPT-2.

    Layer norm normalizes activations across the feature dimension,
    stabilizing training and allowing deeper networks.
    """

    def __init__(self, dim: DimLike, *, eps: float = 1e-5):
        """Initialize layer normalization.

        Args:
            dim: Dimension to normalize (embedding dimension)
            eps: Small epsilon for numerical stability
        """
        super().__init__()
        self.eps = eps
        # Learnable scale parameter (gamma)
        self.weight = Tensor.ones([dim])
        # Learnable shift parameter (beta)
        self.bias = Tensor.zeros([dim])

    def __call__(self, x: Tensor) -> Tensor:
        """Apply layer normalization.

        Args:
            x: Input tensor, shape [..., dim]

        Returns:
            Normalized tensor, same shape as input
        """
        return F.layer_norm(x, gamma=self.weight, beta=self.bias, epsilon=self.eps)


class ResidualBlock(Module):
    """Demonstrates residual connections with layer normalization.

    This shows the pre-norm architecture used in GPT-2:
    output = input + sublayer(layer_norm(input))
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        """Initialize residual block.

        Args:
            dim: Dimension of the input/output
            eps: Epsilon for layer normalization
        """
        super().__init__()
        self.ln = LayerNorm(dim, eps=eps)

    def __call__(self, x: Tensor, sublayer_output: Tensor) -> Tensor:
        """Apply residual connection.

        This demonstrates the pattern:
        1. Normalize input: ln(x)
        2. Apply sublayer (passed as argument for simplicity)
        3. Add residual: x + sublayer_output

        In practice, the sublayer (attention or MLP) is applied to ln(x),
        but we receive the result as a parameter for clarity.

        Args:
            x: Input tensor (the residual)
            sublayer_output: Output from sublayer applied to ln(x)

        Returns:
            x + sublayer_output
        """
        # In a real transformer block, you would do:
        # residual = x
        # x = self.ln(x)
        # x = sublayer(x)  # e.g., attention or MLP
        # x = x + residual

        # For this demonstration, we just add
        return x + sublayer_output


def apply_residual_connection(input_tensor: Tensor, sublayer_output: Tensor) -> Tensor:
    """Apply a residual connection by adding input to sublayer output.

    Residual connections allow gradients to flow directly through the network,
    enabling training of very deep models.

    Args:
        input_tensor: Original input (the residual)
        sublayer_output: Output from a sublayer (attention, MLP, etc.)

    Returns:
        input_tensor + sublayer_output
    """
    return input_tensor + sublayer_output

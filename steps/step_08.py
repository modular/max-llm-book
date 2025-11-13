"""
Step 08: Residual Connections and Layer Normalization

Implement layer normalization and residual connections, which enable
training deep transformer networks by stabilizing gradients.

Tasks:
1. Import F (functional), Tensor, DimLike, and Module
2. Create LayerNorm class with learnable weight and bias parameters
3. Implement layer norm using F.layer_norm
4. Implement residual connection (simple addition)

Run: pixi run s08
"""

# TODO: Import required modules
# Hint: You'll need F from max.experimental
# Hint: You'll need Tensor from max.experimental.tensor
# Hint: You'll need DimLike from max.graph
# Hint: You'll need Module from max.nn.module_v3


class LayerNorm(Module):
    """Layer normalization module matching HuggingFace GPT-2."""

    def __init__(self, dim: DimLike, *, eps: float = 1e-5):
        """Initialize layer normalization.

        Args:
            dim: Dimension to normalize (embedding dimension)
            eps: Small epsilon for numerical stability
        """
        super().__init__()
        self.eps = eps

        # TODO: Create learnable scale parameter (weight)
        # Hint: Use Tensor.ones([dim])
        self.weight = None

        # TODO: Create learnable shift parameter (bias)
        # Hint: Use Tensor.zeros([dim])
        self.bias = None

    def __call__(self, x: Tensor) -> Tensor:
        """Apply layer normalization.

        Args:
            x: Input tensor, shape [..., dim]

        Returns:
            Normalized tensor, same shape as input
        """
        # TODO: Apply layer normalization
        # Hint: Use F.layer_norm(x, gamma=self.weight, beta=self.bias, epsilon=self.eps)
        return None


class ResidualBlock(Module):
    """Demonstrates residual connections with layer normalization."""

    def __init__(self, dim: int, eps: float = 1e-5):
        """Initialize residual block.

        Args:
            dim: Dimension of the input/output
            eps: Epsilon for layer normalization
        """
        super().__init__()

        # TODO: Create layer normalization
        # Hint: Use LayerNorm(dim, eps=eps)
        self.ln = None

    def __call__(self, x: Tensor, sublayer_output: Tensor) -> Tensor:
        """Apply residual connection.

        Args:
            x: Input tensor (the residual)
            sublayer_output: Output from sublayer applied to ln(x)

        Returns:
            x + sublayer_output
        """
        # TODO: Add input and sublayer output (residual connection)
        # Hint: return x + sublayer_output
        return None


def apply_residual_connection(input_tensor: Tensor, sublayer_output: Tensor) -> Tensor:
    """Apply a residual connection by adding input to sublayer output.

    Args:
        input_tensor: Original input (the residual)
        sublayer_output: Output from a sublayer (attention, MLP, etc.)

    Returns:
        input_tensor + sublayer_output
    """
    # TODO: Add the two tensors
    # Hint: return input_tensor + sublayer_output
    return None

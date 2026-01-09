"""
Step 03: Causal Masking

Implement causal attention masking that prevents tokens from attending to future positions.

Tasks:
1. Import functional module (as F) and Tensor from max.experimental
2. Add @F.functional decorator to the causal_mask function
3. Create a constant tensor filled with negative infinity
4. Broadcast the mask to the correct shape (sequence_length, n)
5. Apply band_part to create the lower triangular causal structure

Run: pixi run s03
"""

# 1: Import the required modules from MAX
from max.driver import Device
from max.dtype import DType
# TODO: Import necessary funcional module from max.experimental with the alias F
# https://docs.modular.com/max/api/python/experimental/functional

# TODO: Import Tensor object from max.experimental.tensor
# https://docs.modular.com/max/api/python/experimental/tensor.Tensor

from max.graph import Dim, DimLike

# 2: Add the @F.functional decorator to make this a MAX functional operation
# TODO: Add the decorator here


def causal_mask(
    sequence_length: DimLike,
    num_tokens: DimLike,
    *,
    dtype: DType,
    device: Device,
):
    """Create a causal mask for autoregressive attention.

    Args:
        sequence_length: Length of the sequence.
        num_tokens: Number of tokens.
        dtype: Data type for the mask.
        device: Device to create the mask on.

    Returns:
        A causal mask tensor.
    """
    # Calculate total sequence length
    n = Dim(sequence_length) + num_tokens

    # 3: Create a constant tensor filled with negative infinity
    # TODO: Use Tensor.constant() with float("-inf"), dtype, and device parameters
    # https://docs.modular.com/max/api/python/experimental/tensor#max.experimental.tensor.Tensor.constant
    # Hint: This creates the base mask value that will block attention to future tokens
    mask = None

    # 4: Broadcast the mask to the correct shape
    # TODO: Use F.broadcast_to() to expand mask to shape (sequence_length, n)
    # https://docs.modular.com/max/api/python/experimental/functional#max.experimental.functional.broadcast_to
    # Hint: This creates a 2D attention mask matrix
    mask = None

    # 5: Apply band_part to create the causal (lower triangular) structure and return the mask
    # TODO: Use F.band_part() with num_lower=None, num_upper=0, exclude=True
    # https://docs.modular.com/max/api/python/experimental/functional/#max.experimental.functional.band_part
    # Hint: This keeps only the lower triangle, allowing attention to past tokens only
    return None

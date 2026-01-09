"""
Solution for Step 03: Causal Masking

This module implements causal attention masking that prevents tokens from
attending to future positions in autoregressive generation.
"""

from max.driver import Device
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.graph import Dim, DimLike


@F.functional
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
    n = Dim(sequence_length) + num_tokens
    mask = Tensor.constant(float("-inf"), dtype=dtype, device=device)
    mask = F.broadcast_to(mask, shape=(sequence_length, n))
    return F.band_part(mask, num_lower=None, num_upper=0, exclude=True)

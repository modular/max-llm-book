# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""
Solution for Step 09: Encode and decode tokens

This module provides utility functions to tokenize input text
and decode token IDs back to text using a tokenizer.
"""

import numpy as np
from max.driver import CPU, Device
from max.dtype import DType
from max.experimental.tensor import Tensor
from transformers import GPT2Tokenizer


def encode_text(
    text: str, tokenizer: GPT2Tokenizer, device: Device, max_length: int = 128
) -> Tensor:
    """Tokenize text and convert to tensor."""
    token_ids = tokenizer.encode(text, max_length=max_length, truncation=True)
    return Tensor.constant([token_ids], dtype=DType.int64, device=device)


def decode_tokens(token_ids: Tensor, tokenizer: GPT2Tokenizer) -> str:
    """Decode token IDs back to text."""
    token_ids_np: np.ndarray = np.from_dlpack(token_ids.to(CPU()))
    if token_ids_np.ndim > 1:
        token_ids_np = token_ids_np.flatten()
    token_ids_list: list = token_ids_np.tolist()
    return tokenizer.decode(token_ids_list, skip_special_tokens=True)

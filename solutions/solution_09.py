"""
Solution for Step 09: Encode and decode tokens

This module provides utility functions to tokenize input text
and decode token IDs back to text using a tokenizer.
"""
import numpy as np
from max.dtype import DType
from max.driver import CPU
from max.experimental.tensor import Tensor


def encode_text(text: str, tokenizer, device, max_length: int = 128):
    """Tokenize text and convert to tensor."""
    token_ids = tokenizer.encode(text, max_length=max_length, truncation=True)
    return Tensor.constant([token_ids], dtype=DType.int64, device=device)

def decode_tokens(token_ids: Tensor, tokenizer):
    """Decode token IDs back to text."""
    token_ids = np.from_dlpack(token_ids.to(CPU()))
    if token_ids.ndim > 1:
        token_ids = token_ids.flatten()
    token_ids = token_ids.tolist()
    return tokenizer.decode(token_ids, skip_special_tokens=True)

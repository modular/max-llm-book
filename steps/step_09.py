"""
Step 09: Encode and decode tokens

This module provides utility functions to tokenize input text
and decode token IDs back to text using a tokenizer.

Tasks:
1. Tokenize text and convert to tensor
2. Decode token IDs back to text

Run: pixi run s09
"""

# TODO: Import required modules
# Hint: You'll need numpy as np
# Hint: You'll need CPU from max.driver
# Hint: You'll need DType from max.dtype
# Hint: You'll need Tensor from max.experimental.tensor


def encode_text(text: str, tokenizer, device, max_length: int = 128):
    """Tokenize text and convert to tensor.

    Args:
        text: Input text to tokenize
        tokenizer: HuggingFace tokenizer
        device: Device to place tensor on
        max_length: Maximum sequence length

    Returns:
        Tensor of token IDs with shape [1, seq_length]
    """
    # TODO: Encode text to token IDs
    # Hint: token_ids = tokenizer.encode(text, max_length=max_length, truncation=True)
    pass

    # TODO: Convert to MAX tensor
    # Hint: return Tensor.constant([token_ids], dtype=DType.int64, device=device)
    # Note: Wrap tokens in a list to create batch dimension
    return None


def decode_tokens(token_ids: Tensor, tokenizer):
    """Decode token IDs back to text.

    Args:
        token_ids: Tensor of token IDs
        tokenizer: HuggingFace tokenizer

    Returns:
        Decoded text string
    """
    # TODO: Convert MAX tensor to NumPy
    # Hint: Move to CPU first with token_ids.to(CPU())
    # Hint: Then convert with np.from_dlpack(...)
    pass

    # TODO: Flatten if needed
    # Hint: if token_ids.ndim > 1: token_ids = token_ids.flatten()
    pass

    # TODO: Convert to Python list
    # Hint: token_ids = token_ids.tolist()
    pass

    # TODO: Decode to text
    # Hint: return tokenizer.decode(token_ids, skip_special_tokens=True)
    return None

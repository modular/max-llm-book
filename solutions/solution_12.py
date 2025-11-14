"""
Solution for Step 12: Text Generation

This module implements autoregressive text generation using the GPT-2 model.
"""

import numpy as np

from max.driver import CPU
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.tensor import Tensor


def generate_next_token(model, input_ids, temperature=1.0, do_sample=True):
    """Generate the next token given input context.

    Args:
        model: GPT-2 model with LM head
        input_ids: Current sequence, shape [batch, seq_length]
        temperature: Sampling temperature (higher = more random)
        do_sample: If True, sample from distribution; if False, use greedy (argmax)

    Returns:
        Next token ID as a Tensor
    """
    # Get logits from model
    logits = model(input_ids)

    # Get logits for last position (next token prediction)
    next_token_logits = logits[0, -1, :]  # Shape: [vocab_size]

    if do_sample and temperature > 0:
        # Apply temperature scaling
        temp_tensor = Tensor.constant(
            temperature, dtype=next_token_logits.dtype, device=next_token_logits.device
        )
        next_token_logits = next_token_logits / temp_tensor

        # Convert to probabilities
        probs = F.softmax(next_token_logits)

        # Sample from distribution
        probs_np = np.from_dlpack(probs.to(CPU()))
        next_token_id = np.random.choice(len(probs_np), p=probs_np)
        next_token_tensor = Tensor.constant(
            next_token_id, dtype=DType.int64, device=input_ids.device
        )
    else:
        # Greedy decoding: select most likely token
        next_token_tensor = F.argmax(next_token_logits)

    return next_token_tensor


def generate_tokens(
    model, input_ids, max_new_tokens=10, temperature=1.0, do_sample=True
):
    """Generate multiple tokens autoregressively.

    Args:
        model: GPT-2 model with LM head
        input_ids: Initial sequence, shape [batch, seq_length]
        max_new_tokens: Number of tokens to generate
        temperature: Sampling temperature
        do_sample: Whether to sample or use greedy decoding

    Returns:
        Generated sequence including input, shape [batch, seq_length + max_new_tokens]
    """
    generated_tokens = input_ids

    for _ in range(max_new_tokens):
        # Generate next token
        next_token = generate_next_token(
            model, generated_tokens, temperature=temperature, do_sample=do_sample
        )

        # Reshape to [1, 1] for concatenation
        next_token_2d = next_token.reshape([1, -1])

        # Append to sequence
        generated_tokens = F.concat([generated_tokens, next_token_2d], axis=1)

    return generated_tokens

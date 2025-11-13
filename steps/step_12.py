"""
Step 12: Text Generation

Implement autoregressive text generation with sampling and temperature control.

Tasks:
1. Import required modules (numpy, F, Tensor, etc.)
2. Implement generate_next_token: get logits, apply temperature, sample/argmax
3. Implement generate_tokens: loop to generate multiple tokens

Run: pixi run s12
"""

# TODO: Import required modules
# Hint: You'll need numpy as np
# Hint: You'll need CPU from max.driver
# Hint: You'll need DType from max.dtype
# Hint: You'll need functional as F from max.experimental
# Hint: You'll need Tensor from max.experimental.tensor


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
    # TODO: Get logits from model
    # Hint: logits = model(input_ids)
    pass

    # TODO: Get logits for last position
    # Hint: next_token_logits = logits[0, -1, :]
    pass

    # TODO: If sampling with temperature
    if do_sample and temperature > 0:
        # TODO: Apply temperature scaling
        # Hint: temp_tensor = Tensor.constant(temperature, dtype=next_token_logits.dtype, device=next_token_logits.device)
        # Hint: next_token_logits = next_token_logits / temp_tensor
        pass

        # TODO: Convert to probabilities
        # Hint: probs = F.softmax(next_token_logits)
        pass

        # TODO: Sample from distribution
        # Hint: probs_np = np.from_dlpack(probs.to(CPU()))
        # Hint: next_token_id = np.random.choice(len(probs_np), p=probs_np)
        # Hint: next_token_tensor = Tensor.constant(next_token_id, dtype=DType.int64, device=input_ids.device)
        pass
    else:
        # TODO: Greedy decoding (select most likely token)
        # Hint: next_token_tensor = F.argmax(next_token_logits)
        pass

    # TODO: Return the next token
    return None


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
    # TODO: Initialize generated tokens with input
    # Hint: generated_tokens = input_ids
    pass

    # TODO: Generation loop
    # Hint: for _ in range(max_new_tokens):
    pass

    # TODO: Generate next token
    # Hint: next_token = generate_next_token(model, generated_tokens, temperature=temperature, do_sample=do_sample)
    pass

    # TODO: Reshape to [1, 1] for concatenation
    # Hint: next_token_2d = next_token.reshape([1, -1])
    pass

    # TODO: Append to sequence
    # Hint: generated_tokens = F.concat([generated_tokens, next_token_2d], axis=1)
    pass

    # TODO: Return generated sequence
    return None

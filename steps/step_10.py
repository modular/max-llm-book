"""
Step 10: Text Generation

Implement autoregressive text generation with sampling and temperature control.

Tasks:
1. Import required modules (numpy, F, Tensor, DType, CPU)
2. Implement the generate_text function with temperature scaling
3. Add sampling logic with temperature control
4. Concatenate new tokens to generate sequences

Run: pixi run s10
"""

# TODO: Import required modules
# Hint: You'll need numpy as np
# Hint: You'll need CPU from max.driver
# Hint: You'll need DType from max.dtype
# Hint: You'll need functional as F from max.experimental
# Hint: You'll need Tensor from max.experimental.tensor

from step_09 import encode_text, decode_tokens


def generate_text(
    model,
    tokenizer,
    device,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 0.8,
    do_sample: bool = True,
):
    """Generate text using the Max model.

    Args:
        model: Compiled MAX model
        tokenizer: HuggingFace tokenizer
        device: Device to run on
        prompt: Starting text
        max_new_tokens: Number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        do_sample: Whether to sample or use greedy decoding

    Returns:
        Generated text string
    """
    # TODO: Tokenize the prompt text
    # Hint: Use encode_text(prompt, tokenizer, device, max_length=100)
    generated_tokens = None

    print(f"Starting generation from: '{prompt}'")
    print(
        f"Settings: max_new_tokens={max_new_tokens}, temperature={temperature}, do_sample={do_sample}"
    )
    print("-" * 50)

    # TODO: Implement generation loop for max_new_tokens steps
    # Hint: for step in range(max_new_tokens):
    pass

    # TODO: Get model predictions (logits) for current sequence
    # Hint: logits = model(generated_tokens)

    # TODO: Extract logits for next token prediction
    # Hint: next_token_logits = logits[0, -1, :]
    # Note: Shape is [batch, seq_len, vocab_size], we want last position

    # TODO: Apply temperature scaling if sampling
    # Hint: if do_sample and temperature > 0:
    #     Create a temperature tensor with Tensor.constant()
    #     Divide next_token_logits by temperature
    #     Apply softmax: probs = F.softmax(next_token_logits)
    #     Convert to numpy: probs_np = np.from_dlpack(probs.to(CPU()))
    #     Sample: next_token_id = np.random.choice(len(probs_np), p=probs_np)
    #     Convert back to tensor: next_token_tensor = Tensor.constant(next_token_id, dtype=DType.int64, device=device)

    # TODO: Use greedy decoding if not sampling
    # Hint: else: next_token_tensor = F.argmax(next_token_logits)

    # TODO: Reshape next token to 2D for concatenation
    # Hint: next_token_2d = next_token_tensor.reshape([1, -1])

    # TODO: Concatenate to growing sequence
    # Hint: generated_tokens = F.concat([generated_tokens, next_token_2d], axis=1)

    # TODO: Print progress every 5 steps
    # Hint: if step % 5 == 0 or step == max_new_tokens - 1:
    #     current_text = decode_tokens(generated_tokens, tokenizer)
    #     print(f"Step {step + 1:2d}: {current_text}")

    # TODO: Decode final generated sequence
    # Hint: final_text = decode_tokens(generated_tokens, tokenizer)
    final_text = None

    print("-" * 50)
    print(f"Final generated text: '{final_text}'")
    return final_text
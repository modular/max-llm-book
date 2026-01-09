"""
Solution for Step 10: Text Generation

This module implements autoregressive text generation using the GPT-2 model.
"""

import numpy as np

from max.driver import CPU
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.tensor import Tensor

from solution_09 import tokenize_text, decode_tokens


def generate_text(
    model,
    tokenizer,
    device,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 0.8,
    do_sample: bool = True,
):
    """Generate text using the Max model."""
    generated_tokens = tokenize_text(prompt, tokenizer, device, max_length=100)

    print(f"Starting generation from: '{prompt}'")
    print(
        f"Settings: max_new_tokens={max_new_tokens}, temperature={temperature}, do_sample={do_sample}"
    )
    print("-" * 50)

    for step in range(max_new_tokens):
        logits = model(generated_tokens)
        next_token_logits = logits[0, -1, :]

        if do_sample and temperature > 0:
            # Simple temperature scaling without top-k
            temp_tensor = Tensor.constant(
                temperature,
                dtype=next_token_logits.dtype,
                device=next_token_logits.device,
            )
            next_token_logits = next_token_logits / temp_tensor
            probs = F.softmax(next_token_logits)

            # Convert to numpy for actual sampling
            probs_np = np.from_dlpack(probs.to(CPU()))
            next_token_id = np.random.choice(len(probs_np), p=probs_np)
            next_token_tensor = Tensor.constant(
                next_token_id, dtype=DType.int64, device=generated_tokens.device
            )
        else:
            next_token_tensor = F.argmax(next_token_logits)

        next_token_2d = next_token_tensor.reshape([1, -1])
        generated_tokens = F.concat([generated_tokens, next_token_2d], axis=1)

        if step % 5 == 0 or step == max_new_tokens - 1:
            current_text = decode_tokens(generated_tokens, tokenizer)
            print(f"Step {step + 1:2d}: {current_text}")

    final_text = decode_tokens(generated_tokens, tokenizer)
    print("-" * 50)
    print(f"Final generated text: '{final_text}'")
    return final_text
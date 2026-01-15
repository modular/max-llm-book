"""
Check for Step 09: Encode and Decode Tokens

Validates that encode_text and decode_tokens functions are correctly implemented.
"""

import sys
import inspect
from pathlib import Path

# Add steps directory to path for imports
steps_dir = Path(__file__).parent.parent / "steps"
sys.path.insert(0, str(steps_dir))


def check_step_09():
    """Validate encode_text and decode_tokens implementation."""
    print("Running checks for Step 09: Encode and Decode Tokens...\n")

    errors = []

    # Import the functions
    try:
        from step_09 import encode_text, decode_tokens
    except ImportError as e:
        print(f"❌ Failed to import functions from step_09: {e}")
        return False

    print("✅ Successfully imported encode_text and decode_tokens")

    # Check 1: Verify encode_text signature
    sig = inspect.signature(encode_text)
    params = list(sig.parameters.keys())
    required_params = ['text', 'tokenizer', 'device']
    for param in required_params:
        if param not in params:
            errors.append(f"encode_text missing required parameter: {param}")
        else:
            print(f"✅ encode_text has parameter: {param}")

    # Check 2: Verify decode_tokens signature
    sig = inspect.signature(decode_tokens)
    params = list(sig.parameters.keys())
    required_params = ['token_ids', 'tokenizer']
    for param in required_params:
        if param not in params:
            errors.append(f"decode_tokens missing required parameter: {param}")
        else:
            print(f"✅ decode_tokens has parameter: {param}")

    # Check 3: Test encode_text
    try:
        from transformers import GPT2Tokenizer
        from max.driver import CPU

        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        device = CPU()

        token_ids = encode_text("Hello, world!", tokenizer, device, max_length=128)

        from max.experimental.tensor import Tensor
        if not isinstance(token_ids, Tensor):
            errors.append(f"encode_text should return a Tensor, got {type(token_ids)}")
        else:
            print(f"✅ encode_text returns a Tensor with shape: {token_ids.shape}")

        # Check shape is 2D [batch, seq_len]
        if len(token_ids.shape) != 2:
            errors.append(f"encode_text should return 2D tensor [batch, seq_len], got shape {token_ids.shape}")

        # Check batch size is 1
        if token_ids.shape[0] != 1:
            errors.append(f"encode_text should return batch_size=1, got {token_ids.shape[0]}")

    except Exception as e:
        errors.append(f"encode_text test failed: {e}")

    # Check 4: Test decode_tokens
    try:
        from transformers import GPT2Tokenizer
        from max.driver import CPU
        from max.dtype import DType
        from max.experimental.tensor import Tensor

        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        device = CPU()

        # Create test tokens
        test_tokens = Tensor.constant([[15496, 11, 995, 0]], dtype=DType.int64, device=device)
        decoded = decode_tokens(test_tokens, tokenizer)

        if not isinstance(decoded, str):
            errors.append(f"decode_tokens should return a string, got {type(decoded)}")
        else:
            print(f"✅ decode_tokens returns a string: '{decoded}'")

    except Exception as e:
        errors.append(f"decode_tokens test failed: {e}")

    # Print results
    print()
    if errors:
        print("❌ ERRORS:")
        for error in errors:
            print(f"  - {error}")
        print()
        return False

    print("🎉 All checks passed for Step 09!")
    return True


if __name__ == "__main__":
    success = check_step_09()
    sys.exit(0 if success else 1)

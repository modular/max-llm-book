"""
Check for Step 04: Multi-head Attention

Validates that GPT2MultiHeadAttention is correctly implemented.
"""

import sys
import inspect
from pathlib import Path

# Add steps directory to path for imports
steps_dir = Path(__file__).parent.parent / "steps"
sys.path.insert(0, str(steps_dir))


def check_step_04():
    """Validate GPT2MultiHeadAttention implementation."""
    print("Running checks for Step 04: Multi-head Attention...\n")

    errors = []

    # Import required modules
    try:
        from step_01 import GPT2Config
        from step_04 import GPT2MultiHeadAttention
    except ImportError as e:
        print(f"❌ Failed to import required modules: {e}")
        return False

    print("✅ Successfully imported GPT2MultiHeadAttention")

    # Check 1: Instantiate the module
    try:
        config = GPT2Config()
        attn = GPT2MultiHeadAttention(config)
        print("✅ GPT2MultiHeadAttention can be instantiated")
    except Exception as e:
        errors.append(f"Failed to instantiate GPT2MultiHeadAttention: {e}")
        print()
        print("❌ ERRORS:")
        for error in errors:
            print(f"  - {error}")
        return False

    # Check 2: Verify required attributes
    required_attrs = ['c_attn', 'c_proj', 'embed_dim', 'num_heads', 'head_dim']
    for attr in required_attrs:
        if not hasattr(attn, attr):
            errors.append(f"GPT2MultiHeadAttention missing attribute: {attr}")
        else:
            print(f"✅ Has attribute: {attr}")

    # Check 3: Verify attribute values
    if hasattr(attn, 'embed_dim') and attn.embed_dim != 768:
        errors.append(f"embed_dim should be 768, got {attn.embed_dim}")

    if hasattr(attn, 'num_heads') and attn.num_heads != 12:
        errors.append(f"num_heads should be 12, got {attn.num_heads}")

    if hasattr(attn, 'head_dim') and attn.head_dim != 64:
        errors.append(f"head_dim should be 64 (768/12), got {attn.head_dim}")

    # Check 4: Verify methods exist
    required_methods = ['_split_heads', '_merge_heads', '_attn', '__call__']
    for method in required_methods:
        if not hasattr(attn, method):
            errors.append(f"GPT2MultiHeadAttention missing method: {method}")
        else:
            print(f"✅ Has method: {method}")

    # Check 5: Try a forward pass
    try:
        from max.dtype import DType
        from max.experimental.tensor import Tensor

        # Create dummy input [batch=1, seq_len=10, n_embd=768]
        dummy_input = Tensor.ones([1, 10, 768], dtype=DType.float32)
        output = attn(dummy_input)

        if output.shape != dummy_input.shape:
            errors.append(f"Output shape mismatch: expected {dummy_input.shape}, got {output.shape}")
        else:
            print(f"✅ Forward pass successful with shape: {output.shape}")
    except Exception as e:
        errors.append(f"Forward pass failed: {e}")

    # Print results
    print()
    if errors:
        print("❌ ERRORS:")
        for error in errors:
            print(f"  - {error}")
        print()
        return False

    print("🎉 All checks passed for Step 04!")
    return True


if __name__ == "__main__":
    success = check_step_04()
    sys.exit(0 if success else 1)

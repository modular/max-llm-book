"""
Check for Step 06: Transformer Block

Validates that GPT2Block is correctly implemented.
"""

import sys
from pathlib import Path

# Add steps directory to path for imports
steps_dir = Path(__file__).parent.parent / "steps"
sys.path.insert(0, str(steps_dir))


def check_step_06():
    """Validate GPT2Block implementation."""
    print("Running checks for Step 06: Transformer Block...\n")

    errors = []

    # Import required modules
    try:
        from step_01 import GPT2Config
        from step_06 import GPT2Block
    except ImportError as e:
        print(f"❌ Failed to import required modules: {e}")
        return False

    print("✅ Successfully imported GPT2Block")

    # Check 1: Instantiate the module
    try:
        config = GPT2Config()
        block = GPT2Block(config)
        print("✅ GPT2Block can be instantiated")
    except Exception as e:
        errors.append(f"Failed to instantiate GPT2Block: {e}")
        print()
        print("❌ ERRORS:")
        for error in errors:
            print(f"  - {error}")
        return False

    # Check 2: Verify required attributes
    required_attrs = ['ln_1', 'attn', 'ln_2', 'mlp']
    for attr in required_attrs:
        if not hasattr(block, attr):
            errors.append(f"GPT2Block missing attribute: {attr}")
        else:
            print(f"✅ Has attribute: {attr}")

    # Check 3: Verify __call__ method
    if not hasattr(block, '__call__'):
        errors.append("GPT2Block missing __call__ method")
    else:
        print("✅ Has __call__ method")

    # Check 4: Try a forward pass
    try:
        from max.dtype import DType
        from max.experimental.tensor import Tensor

        dummy_input = Tensor.ones([1, 10, 768], dtype=DType.float32)
        output = block(dummy_input)

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

    print("🎉 All checks passed for Step 06!")
    return True


if __name__ == "__main__":
    success = check_step_06()
    sys.exit(0 if success else 1)

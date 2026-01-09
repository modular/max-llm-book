"""
Check for Step 08: Language Model Head

Validates that MaxGPT2LMHeadModel is correctly implemented.
"""

import sys
from pathlib import Path

# Add steps directory to path for imports
steps_dir = Path(__file__).parent.parent / "steps"
sys.path.insert(0, str(steps_dir))


def check_step_08():
    """Validate MaxGPT2LMHeadModel implementation."""
    print("Running checks for Step 08: Language Model Head...\n")

    errors = []

    # Import required modules
    try:
        from step_01 import GPT2Config
        from step_08 import MaxGPT2LMHeadModel
    except ImportError as e:
        print(f"❌ Failed to import required modules: {e}")
        return False

    print("✅ Successfully imported MaxGPT2LMHeadModel")

    # Check 1: Instantiate the module
    try:
        config = GPT2Config()
        model = MaxGPT2LMHeadModel(config)
        print("✅ MaxGPT2LMHeadModel can be instantiated")
    except Exception as e:
        errors.append(f"Failed to instantiate MaxGPT2LMHeadModel: {e}")
        print()
        print("❌ ERRORS:")
        for error in errors:
            print(f"  - {error}")
        return False

    # Check 2: Verify required attributes
    required_attrs = ['transformer', 'lm_head', 'config']
    for attr in required_attrs:
        if not hasattr(model, attr):
            errors.append(f"MaxGPT2LMHeadModel missing attribute: {attr}")
        else:
            print(f"✅ Has attribute: {attr}")

    # Check 3: Verify lm_head is Linear
    if hasattr(model, 'lm_head'):
        from max.nn.module_v3 import Linear
        if not isinstance(model.lm_head, Linear):
            errors.append(f"lm_head should be Linear, got {type(model.lm_head)}")

    # Check 4: Verify __call__ method
    if not hasattr(model, '__call__'):
        errors.append("MaxGPT2LMHeadModel missing __call__ method")
    else:
        print("✅ Has __call__ method")

    # Check 5: Try a forward pass
    try:
        from max.dtype import DType
        from max.experimental.tensor import Tensor

        # Create dummy token IDs [batch=1, seq_len=10]
        dummy_input = Tensor.constant([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=DType.int64)
        output = model(dummy_input)

        expected_shape = (1, 10, 50257)  # [batch, seq_len, vocab_size]
        actual_shape = tuple(int(dim) for dim in output.shape)
        if actual_shape != expected_shape:
            errors.append(f"Output shape mismatch: expected {expected_shape}, got {actual_shape}")
        else:
            print(f"✅ Forward pass successful with shape: {actual_shape}")
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

    print("🎉 All checks passed for Step 08!")
    return True


if __name__ == "__main__":
    success = check_step_08()
    sys.exit(0 if success else 1)

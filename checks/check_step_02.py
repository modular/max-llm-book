"""
Check for Step 02: Feed-forward Network (MLP)

Validates that GPT2MLP is correctly implemented with proper structure.
"""

import sys
import inspect
from pathlib import Path

# Add steps directory to path for imports
steps_dir = Path(__file__).parent.parent / "steps"
sys.path.insert(0, str(steps_dir))


def check_step_02():
    """Validate GPT2MLP implementation."""
    print("Running checks for Step 02: Feed-forward Network (MLP)...\n")

    errors = []

    # Import required modules
    try:
        from step_01 import GPT2Config
        from step_02 import GPT2MLP
    except ImportError as e:
        print(f"❌ Failed to import required modules: {e}")
        return False

    print("✅ Successfully imported GPT2MLP and GPT2Config")

    # Check 1: Can we instantiate GPT2MLP?
    try:
        config = GPT2Config()
        mlp = GPT2MLP(intermediate_size=3072, config=config)
        print("✅ GPT2MLP can be instantiated")
    except Exception as e:
        errors.append(f"Failed to instantiate GPT2MLP: {e}")
        print()
        print("❌ ERRORS:")
        for error in errors:
            print(f"  - {error}")
        return False

    # Check 2: Verify it has required attributes
    required_attrs = ['c_fc', 'c_proj']
    for attr in required_attrs:
        if not hasattr(mlp, attr):
            errors.append(f"GPT2MLP missing required attribute: {attr}")
        else:
            print(f"✅ GPT2MLP has attribute: {attr}")

    # Check 3: Verify __call__ method exists and has correct signature
    if not hasattr(mlp, '__call__'):
        errors.append("GPT2MLP missing __call__ method")
    else:
        sig = inspect.signature(mlp.__call__)
        params = list(sig.parameters.keys())
        if 'hidden_states' not in params:
            errors.append("__call__ method should accept 'hidden_states' parameter")
        else:
            print("✅ GPT2MLP has __call__ method with correct signature")

    # Check 4: Try a forward pass with dummy tensor
    try:
        from max.dtype import DType
        from max.experimental.tensor import Tensor

        # Create dummy input [batch=1, seq_len=10, n_embd=768]
        dummy_input = Tensor.ones([1, 10, 768], dtype=DType.float32)
        output = mlp(dummy_input)

        # Check output shape
        if output.shape != dummy_input.shape:
            errors.append(
                f"Output shape mismatch: expected {dummy_input.shape}, "
                f"got {output.shape}"
            )
        else:
            print(f"✅ Forward pass successful with correct output shape: {output.shape}")
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

    print("🎉 All checks passed for Step 02!")
    return True


if __name__ == "__main__":
    success = check_step_02()
    sys.exit(0 if success else 1)

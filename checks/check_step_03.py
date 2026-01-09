"""
Check for Step 03: Causal Masking

Validates that causal_mask function is correctly implemented.
"""

import sys
import inspect
from pathlib import Path

# Add steps directory to path for imports
steps_dir = Path(__file__).parent.parent / "steps"
sys.path.insert(0, str(steps_dir))


def check_step_03():
    """Validate causal_mask implementation."""
    print("Running checks for Step 03: Causal Masking...\n")

    errors = []

    # Import the function
    try:
        from step_03 import causal_mask
    except ImportError as e:
        print(f"❌ Failed to import causal_mask from step_03: {e}")
        return False

    print("✅ Successfully imported causal_mask")

    # Check 1: Verify function signature
    sig = inspect.signature(causal_mask)
    params = list(sig.parameters.keys())
    required_params = ['sequence_length', 'num_tokens']
    for param in required_params:
        if param not in params:
            errors.append(f"causal_mask missing required parameter: {param}")
        else:
            print(f"✅ causal_mask has parameter: {param}")

    # Check 2: Verify keyword-only parameters
    kw_only_params = ['dtype', 'device']
    for param in kw_only_params:
        if param not in params:
            errors.append(f"causal_mask missing keyword parameter: {param}")

    # Check 3: Test function call
    try:
        from max.driver import CPU
        from max.dtype import DType

        # Create a simple causal mask
        mask = causal_mask(5, 0, dtype=DType.float32, device=CPU())

        # Check output type
        from max.experimental.tensor import Tensor
        if not isinstance(mask, Tensor):
            errors.append(f"causal_mask should return a Tensor, got {type(mask)}")
        else:
            print(f"✅ causal_mask returns a Tensor")

        # Check shape
        expected_shape = (5, 5)
        actual_shape = tuple(int(dim) for dim in mask.shape)
        if actual_shape != expected_shape:
            errors.append(f"Expected mask shape {expected_shape}, got {actual_shape}")
        else:
            print(f"✅ Mask has correct shape: {actual_shape}")

    except Exception as e:
        errors.append(f"Failed to call causal_mask: {e}")

    # Print results
    print()
    if errors:
        print("❌ ERRORS:")
        for error in errors:
            print(f"  - {error}")
        print()
        return False

    print("🎉 All checks passed for Step 03!")
    return True


if __name__ == "__main__":
    success = check_step_03()
    sys.exit(0 if success else 1)

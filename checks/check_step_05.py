"""
Check for Step 05: Layer Normalization

Validates that LayerNorm is correctly implemented.
"""

import sys
import inspect


def check_step_05():
    """Validate LayerNorm implementation."""
    print("Running checks for Step 05: Layer Normalization...\n")

    errors = []

    # Import the module
    try:
        from step_05 import LayerNorm
    except ImportError as e:
        print(f"❌ Failed to import LayerNorm from step_05: {e}")
        return False

    print("✅ Successfully imported LayerNorm")

    # Check 1: Instantiate the module
    try:
        ln = LayerNorm(768, eps=1e-5)
        print("✅ LayerNorm can be instantiated")
    except Exception as e:
        errors.append(f"Failed to instantiate LayerNorm: {e}")
        print()
        print("❌ ERRORS:")
        for error in errors:
            print(f"  - {error}")
        return False

    # Check 2: Verify required attributes
    required_attrs = ['weight', 'bias', 'eps']
    for attr in required_attrs:
        if not hasattr(ln, attr):
            errors.append(f"LayerNorm missing attribute: {attr}")
        else:
            print(f"✅ Has attribute: {attr}")

    # Check 3: Verify weight and bias shapes
    if hasattr(ln, 'weight'):
        from max.experimental.tensor import Tensor
        if not isinstance(ln.weight, Tensor):
            errors.append(f"weight should be a Tensor, got {type(ln.weight)}")
        elif ln.weight.shape != (768,):
            errors.append(f"weight shape should be (768,), got {ln.weight.shape}")
        else:
            print(f"✅ weight has correct shape: {ln.weight.shape}")

    if hasattr(ln, 'bias'):
        from max.experimental.tensor import Tensor
        if not isinstance(ln.bias, Tensor):
            errors.append(f"bias should be a Tensor, got {type(ln.bias)}")
        elif ln.bias.shape != (768,):
            errors.append(f"bias shape should be (768,), got {ln.bias.shape}")
        else:
            print(f"✅ bias has correct shape: {ln.bias.shape}")

    # Check 4: Verify __call__ method
    if not hasattr(ln, '__call__'):
        errors.append("LayerNorm missing __call__ method")
    else:
        print("✅ Has __call__ method")

    # Check 5: Try a forward pass
    try:
        from max.dtype import DType
        from max.experimental.tensor import Tensor

        dummy_input = Tensor.ones([1, 10, 768], dtype=DType.float32)
        output = ln(dummy_input)

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

    print("🎉 All checks passed for Step 05!")
    return True


if __name__ == "__main__":
    success = check_step_05()
    sys.exit(0 if success else 1)

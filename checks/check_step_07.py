"""
Check for Step 07: Stacking Transformer Blocks

Validates that MaxGPT2Model is correctly implemented.
"""

import sys
from pathlib import Path

# Add steps directory to path for imports
steps_dir = Path(__file__).parent.parent / "steps"
sys.path.insert(0, str(steps_dir))


def check_step_07():
    """Validate MaxGPT2Model implementation."""
    print("Running checks for Step 07: Stacking Transformer Blocks...\n")

    errors = []

    # Import required modules
    try:
        from step_01 import GPT2Config
        from step_07 import MaxGPT2Model
    except ImportError as e:
        print(f"❌ Failed to import required modules: {e}")
        return False

    print("✅ Successfully imported MaxGPT2Model")

    # Check 1: Instantiate the module
    try:
        config = GPT2Config()
        model = MaxGPT2Model(config)
        print("✅ MaxGPT2Model can be instantiated")
    except Exception as e:
        errors.append(f"Failed to instantiate MaxGPT2Model: {e}")
        print()
        print("❌ ERRORS:")
        for error in errors:
            print(f"  - {error}")
        return False

    # Check 2: Verify required attributes
    required_attrs = ['wte', 'wpe', 'h', 'ln_f']
    for attr in required_attrs:
        if not hasattr(model, attr):
            errors.append(f"MaxGPT2Model missing attribute: {attr}")
        else:
            print(f"✅ Has attribute: {attr}")

    # Check 3: Verify embeddings
    if hasattr(model, 'wte'):
        from max.nn.module_v3 import Embedding
        if not isinstance(model.wte, Embedding):
            errors.append(f"wte should be an Embedding, got {type(model.wte)}")

    if hasattr(model, 'wpe'):
        from max.nn.module_v3 import Embedding
        if not isinstance(model.wpe, Embedding):
            errors.append(f"wpe should be an Embedding, got {type(model.wpe)}")

    # Check 4: Verify __call__ method
    if not hasattr(model, '__call__'):
        errors.append("MaxGPT2Model missing __call__ method")
    else:
        print("✅ Has __call__ method")

    # Check 5: Try a forward pass
    try:
        from max.dtype import DType
        from max.experimental.tensor import Tensor

        # Create dummy token IDs [batch=1, seq_len=10]
        dummy_input = Tensor.constant([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=DType.int64)
        output = model(dummy_input)

        expected_shape = (1, 10, 768)  # [batch, seq_len, n_embd]
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

    print("🎉 All checks passed for Step 07!")
    return True


if __name__ == "__main__":
    success = check_step_07()
    sys.exit(0 if success else 1)

"""
Check for Step 01: Model Configuration

Validates that GPT2Config dataclass is correctly implemented with proper values.
"""

import sys
from pathlib import Path
from dataclasses import is_dataclass

# Add steps directory to path for imports
steps_dir = Path(__file__).parent.parent / "steps"
sys.path.insert(0, str(steps_dir))


def check_step_01():
    """Validate GPT2Config implementation."""
    print("Running checks for Step 01: Model Configuration...\n")

    errors = []
    warnings = []

    # Import the student's implementation
    try:
        from step_01 import GPT2Config
    except ImportError as e:
        print(f"❌ Failed to import GPT2Config from step_01: {e}")
        return False

    # Check 1: Is it a dataclass?
    if not is_dataclass(GPT2Config):
        errors.append("GPT2Config must be a dataclass (use @dataclass decorator)")
    else:
        print("✅ GPT2Config is a dataclass")

    # Check 2: Can we instantiate it with defaults?
    try:
        config = GPT2Config()
        print("✅ GPT2Config can be instantiated with default values")
    except Exception as e:
        errors.append(f"Failed to instantiate GPT2Config: {e}")
        print()
        if errors:
            print("❌ ERRORS:")
            for error in errors:
                print(f"  - {error}")
        return False

    # Check 3: Validate field values match HuggingFace GPT-2
    expected_values = {
        'vocab_size': 50257,
        'n_positions': 1024,
        'n_embd': 768,
        'n_layer': 12,
        'n_head': 12,
        'n_inner': 3072,  # 4 * n_embd
        'layer_norm_epsilon': 1e-5,
    }

    for field_name, expected_value in expected_values.items():
        if not hasattr(config, field_name):
            errors.append(f"Missing field: {field_name}")
        else:
            actual_value = getattr(config, field_name)
            if actual_value != expected_value:
                errors.append(
                    f"Field '{field_name}' has incorrect value: "
                    f"expected {expected_value}, got {actual_value}"
                )
            else:
                print(f"✅ {field_name} = {actual_value}")

    # Check 4: Validate field types
    expected_types = {
        'vocab_size': int,
        'n_positions': int,
        'n_embd': int,
        'n_layer': int,
        'n_head': int,
        'n_inner': int,
        'layer_norm_epsilon': float,
    }

    for field_name, expected_type in expected_types.items():
        if hasattr(config, field_name):
            actual_value = getattr(config, field_name)
            if not isinstance(actual_value, expected_type):
                warnings.append(
                    f"Field '{field_name}' should be {expected_type.__name__}, "
                    f"got {type(actual_value).__name__}"
                )

    # Print results
    print()
    if errors:
        print("❌ ERRORS:")
        for error in errors:
            print(f"  - {error}")
        print()
        return False

    if warnings:
        print("⚠️  WARNINGS:")
        for warning in warnings:
            print(f"  - {warning}")
        print()

    print("🎉 All checks passed for Step 01!")
    return True


if __name__ == "__main__":
    success = check_step_01()
    sys.exit(0 if success else 1)

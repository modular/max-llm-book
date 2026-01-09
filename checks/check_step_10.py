"""
Check for Step 10: Text Generation

Validates that generate_text function is correctly implemented.
"""

import sys
import inspect
from pathlib import Path

# Add steps directory to path for imports
steps_dir = Path(__file__).parent.parent / "steps"
sys.path.insert(0, str(steps_dir))


def check_step_10():
    """Validate generate_text implementation."""
    print("Running checks for Step 10: Text Generation...\n")

    errors = []

    # Import the function
    try:
        from step_10 import generate_text
    except ImportError as e:
        print(f"❌ Failed to import generate_text from step_10: {e}")
        return False

    print("✅ Successfully imported generate_text")

    # Check 1: Verify function signature
    sig = inspect.signature(generate_text)
    params = list(sig.parameters.keys())
    required_params = ['model', 'tokenizer', 'device', 'prompt']
    for param in required_params:
        if param not in params:
            errors.append(f"generate_text missing required parameter: {param}")
        else:
            print(f"✅ generate_text has parameter: {param}")

    # Check 2: Verify optional parameters with defaults
    optional_params = {
        'max_new_tokens': 50,
        'temperature': 0.8,
        'do_sample': True
    }
    for param, default_value in optional_params.items():
        if param in params:
            param_obj = sig.parameters[param]
            if param_obj.default != inspect.Parameter.empty:
                print(f"✅ generate_text has optional parameter: {param} (default={param_obj.default})")
            else:
                errors.append(f"Parameter '{param}' should have a default value")
        else:
            errors.append(f"generate_text missing optional parameter: {param}")

    # Check 3: Verify return type annotation (if present)
    if sig.return_annotation != inspect.Signature.empty:
        if sig.return_annotation == str:
            print("✅ generate_text annotated to return str")

    # Note: We cannot easily test the actual generation without a compiled model,
    # so we just verify the function signature and structure

    # Print results
    print()
    if errors:
        print("❌ ERRORS:")
        for error in errors:
            print(f"  - {error}")
        print()
        return False

    print("🎉 All checks passed for Step 10!")
    print("Note: Full generation testing requires a compiled model")
    return True


if __name__ == "__main__":
    success = check_step_10()
    sys.exit(0 if success else 1)

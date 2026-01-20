"""
Check for Step 11: Load Weights and Run Model

Validates that run_model function is correctly implemented.
"""

import sys
import inspect
from pathlib import Path

# Add steps directory to path for imports
steps_dir = Path(__file__).parent.parent / "steps"
sys.path.insert(0, str(steps_dir))


def check_step_11():
    """Validate run_model implementation."""
    print("Running checks for Step 11: Load Weights and Run Model...\n")

    errors = []

    # Import the function
    try:
        from step_11 import run_model
    except ImportError as e:
        print(f"❌ Failed to import run_model from step_11: {e}")
        return False

    print("✅ Successfully imported run_model")

    # Check 1: Verify function exists and is callable
    if not callable(run_model):
        errors.append("run_model should be a callable function")
    else:
        print("✅ run_model is callable")

    # Check 2: Verify function signature
    sig = inspect.signature(run_model)
    # run_model should take no required arguments
    required_params = [p for p in sig.parameters.values() if p.default == inspect.Parameter.empty]
    if len(required_params) > 0:
        errors.append(f"run_model should not have required parameters, found: {[p.name for p in required_params]}")
    else:
        print("✅ run_model has correct signature (no required parameters)")

    # Check 3: Verify the function contains key implementation steps
    # We can't run it without user interaction, but we can check the source
    try:
        import inspect as insp
        source = insp.getsource(run_model)

        # Check for key components in the source
        required_components = [
            ('GPT2LMHeadModel', 'Loading HuggingFace model'),
            ('GPT2Tokenizer', 'Initializing tokenizer'),
            ('from_pretrained', 'Using from_pretrained method'),
            ('compile', 'Compiling the model'),
            ('generate_text', 'Calling generate_text function'),
        ]

        for component, description in required_components:
            if component in source:
                print(f"✅ Function contains: {description}")
            else:
                errors.append(f"Function missing: {description} ('{component}' not found in source)")

    except Exception as e:
        errors.append(f"Failed to analyze function source: {e}")

    # Print results
    print()
    if errors:
        print("❌ ERRORS:")
        for error in errors:
            print(f"  - {error}")
        print()
        return False

    print("🎉 All checks passed for Step 11!")
    print("Note: Full testing requires running the interactive model (use: pixi run gpt2)")
    return True


if __name__ == "__main__":
    success = check_step_11()
    sys.exit(0 if success else 1)

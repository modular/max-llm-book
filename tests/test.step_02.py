"""Tests for causal_mask implementation in steps/step_02.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import inspect


def test_step_02():
    """Test that causal_mask function is correctly implemented."""
    print("Running tests for Step 02: Implement Causal Masking...\n")
    print("Results:")

    # Test 1: Check if functional module is imported
    try:
        from steps import step_02 as mask_module

        # Check if F is imported from max.experimental.functional
        source = inspect.getsource(mask_module)
        if "from max.experimental import functional as F" in source:
            print(
                "✅ functional module is correctly imported as F from max.experimental"
            )
        else:
            print("❌ functional module is not imported from max.experimental")
            print("   Hint: Add 'from max.experimental import functional as F'")
    except Exception as e:
        print(f"❌ Error importing step_02 module: {e}")
        return

    # Test 2: Check if Tensor is imported
    if "from max.experimental.tensor import Tensor" in source:
        print("✅ Tensor is correctly imported from max.experimental.tensor")
    else:
        print("❌ Tensor is not imported from max.experimental.tensor")
        print("   Hint: Add 'from max.experimental.tensor import Tensor'")

    # Test 3: Check if causal_mask function exists and has @F.functional decorator
    if hasattr(mask_module, "causal_mask"):
        # Check for decorator by looking at source
        func_source = inspect.getsource(mask_module.causal_mask)
        if "@F.functional" in func_source:
            print("✅ causal_mask function has @F.functional decorator")
        else:
            print("❌ causal_mask function does not have @F.functional decorator")
            print(
                "   Hint: Add '@F.functional' decorator above the function definition"
            )
    else:
        print("❌ causal_mask function not found in step_02 module")
        return

    # Test 4: Check if Tensor.constant is used
    if "Tensor.constant" in source and 'float("-inf")' in source:
        print("✅ Tensor.constant is used with float('-inf')")
    else:
        print("❌ Tensor.constant is not used correctly")
        print('   Hint: Use Tensor.constant(float("-inf"), dtype=dtype, device=device)')

    # Test 5: Check if F.broadcast_to is used
    if "F.broadcast_to" in source and "shape=(sequence_length, n)" in source:
        print("✅ F.broadcast_to is used with correct shape")
    else:
        print("❌ F.broadcast_to is not used correctly")
        print("   Hint: Use F.broadcast_to(mask, shape=(sequence_length, n))")

    # Test 6: Check if F.band_part is used with correct parameters
    if "F.band_part" in source:
        if (
            "num_lower=None" in source
            and "num_upper=0" in source
            and "exclude=True" in source
        ):
            print(
                "✅ F.band_part is used with correct parameters (num_lower=None, num_upper=0, exclude=True)"
            )
        else:
            print("❌ F.band_part is not used with correct parameters")
            print(
                "   Hint: Use F.band_part(mask, num_lower=None, num_upper=0, exclude=True)"
            )
    else:
        print("❌ F.band_part is not used")
        print("   Hint: Use F.band_part to create the lower triangular mask")

    # Test 7: Check that None values are replaced
    lines = source.split("\n")
    none_assignments = [
        line for line in lines if "mask = None" in line or "return None" in line
    ]

    if none_assignments:
        print("❌ Found placeholder 'None' values that need to be replaced:")
        for line in none_assignments:
            print(f"   {line.strip()}")
        print("   Hint: Replace all 'None' values with the actual implementation")
    else:
        print("✅ All placeholder 'None' values have been replaced")

    # Test 8: Try to run the function (if imports are available)
    try:
        from max.driver import CPU
        from max.dtype import DType

        # Try to create a simple mask
        result = mask_module.causal_mask(
            sequence_length=4, num_tokens=2, dtype=DType.float32, device=CPU()
        )

        print("✅ causal_mask function executes without errors")

        # Check shape
        expected_shape = (4, 6)  # (sequence_length, sequence_length + num_tokens)
        if hasattr(result, "shape"):
            actual_shape = tuple(result.shape)
            if actual_shape == expected_shape:
                print(f"✅ Mask shape is correct: {actual_shape}")
            else:
                print(
                    f"❌ Mask shape is incorrect: expected {expected_shape}, got {actual_shape}"
                )

    except ImportError:
        print(
            "⚠️  Cannot test function execution (MAX not available in this environment)"
        )
    except AttributeError as e:
        print(f"❌ Function execution failed: {e}")
        print("   This usually means some TODO items are not completed")
    except Exception as e:
        print(f"❌ Function execution failed with error: {e}")

    # Final summary
    print("\n" + "=" * 60)
    if all(
        [
            "from max.experimental import functional as F" in source,
            "from max.experimental.tensor import Tensor" in source,
            "@F.functional" in source,
            "Tensor.constant" in source,
            "F.broadcast_to" in source,
            "F.band_part" in source,
            not none_assignments,
        ]
    ):
        print("🎉 All checks passed! Your implementation matches the solution.")
        print("=" * 60)
    else:
        print("⚠️  Some checks failed. Review the hints above and try again.")
        print("=" * 60)


if __name__ == "__main__":
    test_step_02()

"""Tests for LayerNorm implementation in steps/step_03.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import inspect


def test_step_03():
    """Test that LayerNorm class is correctly implemented."""
    print("Running tests for Step 03: Implement Layer Normalization...\n")
    print("Results:")

    # Test 1: Check if functional module is imported
    try:
        from steps import step_03 as layer_norm_module

        # Check if F is imported from max.experimental.functional
        source = inspect.getsource(layer_norm_module)
        if "from max.experimental import functional as F" in source:
            print(
                "✅ functional module is correctly imported as F from max.experimental"
            )
        else:
            print("❌ functional module is not imported from max.experimental")
            print("   Hint: Add 'from max.experimental import functional as F'")
    except Exception as e:
        print(f"❌ Error importing step_03 module: {e}")
        return

    # Test 2: Check if Tensor is imported
    if "from max.experimental.tensor import Tensor" in source:
        print("✅ Tensor is correctly imported from max.experimental.tensor")
    else:
        print("❌ Tensor is not imported from max.experimental.tensor")
        print("   Hint: Add 'from max.experimental.tensor import Tensor'")

    # Test 3: Check if LayerNorm class exists
    if hasattr(layer_norm_module, "LayerNorm"):
        print("✅ LayerNorm class exists")
    else:
        print("❌ LayerNorm class not found in step_03 module")
        return

    # Test 4: Check if Tensor.ones is used for weight initialization
    if (
        "self.weight = Tensor.ones([dim])" in source
        or "self.weight = Tensor.ones([" in source
    ):
        print("✅ Tensor.ones is used for weight initialization")
    else:
        print("❌ Tensor.ones is not used correctly for weight initialization")
        print("   Hint: Use Tensor.ones([dim]) to initialize self.weight")

    # Test 5: Check if Tensor.zeros is used for bias initialization
    if (
        "self.bias = Tensor.zeros([dim])" in source
        or "self.bias = Tensor.zeros([" in source
    ):
        print("✅ Tensor.zeros is used for bias initialization")
    else:
        print("❌ Tensor.zeros is not used correctly for bias initialization")
        print("   Hint: Use Tensor.zeros([dim]) to initialize self.bias")

    # Test 6: Check if F.layer_norm is used
    if "F.layer_norm" in source:
        print("✅ F.layer_norm is used")
    else:
        print("❌ F.layer_norm is not used")
        print("   Hint: Use F.layer_norm() in the __call__ method")

    # Test 7: Check if F.layer_norm has correct parameters
    if (
        "gamma=self.weight" in source
        and "beta=self.bias" in source
        and "epsilon=self.eps" in source
    ):
        print("✅ F.layer_norm is used with correct parameters (gamma, beta, epsilon)")
    else:
        print("❌ F.layer_norm is not used with correct parameters")
        print(
            "   Hint: Use F.layer_norm(x, gamma=self.weight, beta=self.bias, epsilon=self.eps)"
        )

    # Test 8: Check that None values are replaced
    lines = source.split("\n")
    none_assignments = [
        line
        for line in lines
        if (
            "self.weight = None" in line
            or "self.bias = None" in line
            or "return None" in line
        )
    ]

    if none_assignments:
        print("❌ Found placeholder 'None' values that need to be replaced:")
        for line in none_assignments:
            print(f"   {line.strip()}")
        print("   Hint: Replace all 'None' values with the actual implementation")
    else:
        print("✅ All placeholder 'None' values have been replaced")

    # Test 9: Try to instantiate the LayerNorm class
    try:
        layer_norm = layer_norm_module.LayerNorm(dim=768)
        print("✅ LayerNorm class can be instantiated")

        # Check if weight and bias are initialized
        if hasattr(layer_norm, "weight") and layer_norm.weight is not None:
            print("✅ LayerNorm.weight is initialized")
        else:
            print("❌ LayerNorm.weight is not initialized")

        if hasattr(layer_norm, "bias") and layer_norm.bias is not None:
            print("✅ LayerNorm.bias is initialized")
        else:
            print("❌ LayerNorm.bias is not initialized")

        if hasattr(layer_norm, "eps"):
            if layer_norm.eps == 1e-5:
                print("✅ LayerNorm.eps has correct default value: 1e-05")
            else:
                print(
                    f"❌ LayerNorm.eps has incorrect value: expected 1e-05, got {layer_norm.eps}"
                )
        else:
            print("❌ LayerNorm.eps is not set")

    except Exception as e:
        print(f"❌ LayerNorm class instantiation failed: {e}")
        print("   This usually means some TODO items are not completed")

    except ImportError:
        print(
            "⚠️  Cannot test layer normalization execution (MAX not available in this environment)"
        )
    except AttributeError as e:
        print(f"❌ Layer normalization execution failed: {e}")
        print("   This usually means some TODO items are not completed")
    except Exception as e:
        print(f"❌ Layer normalization execution failed with error: {e}")

    # Final summary
    print("\n" + "=" * 60)
    if all(
        [
            "from max.experimental import functional as F" in source,
            "from max.experimental.tensor import Tensor" in source,
            "Tensor.ones" in source,
            "Tensor.zeros" in source,
            "F.layer_norm" in source,
            "gamma=self.weight" in source,
            "beta=self.bias" in source,
            "epsilon=self.eps" in source,
            not none_assignments,
        ]
    ):
        print("🎉 All checks passed! Your implementation matches the solution.")
        print("=" * 60)
    else:
        print("⚠️  Some checks failed. Review the hints above and try again.")
        print("=" * 60)


if __name__ == "__main__":
    test_step_03()

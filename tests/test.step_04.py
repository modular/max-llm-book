"""Tests for GPT2MLP implementation in steps/step_04.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import inspect


def test_step_04():
    """Test that GPT2MLP class is correctly implemented."""
    print("Running tests for Step 04: Implement GPT-2 MLP (Feed-Forward Network)...\n")
    print("Results:")

    # Test 1: Check if functional module is imported
    try:
        from steps import step_04 as mlp_module

        # Check if F is imported from max.experimental.functional
        source = inspect.getsource(mlp_module)
        if "from max.experimental import functional as F" in source:
            print(
                "✅ functional module is correctly imported as F from max.experimental"
            )
        else:
            print("❌ functional module is not imported from max.experimental")
            print("   Hint: Add 'from max.experimental import functional as F'")
    except Exception as e:
        print(f"❌ Error importing step_04 module: {e}")
        return

    # Test 2: Check if Tensor is imported
    if "from max.experimental.tensor import Tensor" in source:
        print("✅ Tensor is correctly imported from max.experimental.tensor")
    else:
        print("❌ Tensor is not imported from max.experimental.tensor")
        print("   Hint: Add 'from max.experimental.tensor import Tensor'")

    # Test 3: Check if Linear and Module are imported
    if (
        "from max.nn.module_v3 import Linear, Module" in source
        or "from max.nn.module_v3 import" in source
    ):
        print("✅ Linear and Module are imported from max.nn.module_v3")
    else:
        print("❌ Linear and Module are not imported from max.nn.module_v3")
        print("   Hint: Add 'from max.nn.module_v3 import Linear, Module'")

    # Test 4: Check if GPT2MLP class exists
    if hasattr(mlp_module, "GPT2MLP"):
        print("✅ GPT2MLP class exists")
    else:
        print("❌ GPT2MLP class not found in step_04 module")
        return

    # Test 5: Check if c_fc Linear layer is created correctly
    if "self.c_fc = Linear(embed_dim, intermediate_size, bias=True)" in source:
        print("✅ self.c_fc Linear layer is created correctly")
    else:
        print("❌  self.c_fc Linear layer is not created correctly")
        print("   Hint: Use Linear(embed_dim, intermediate_size, bias=True)")

    # Test 6: Check if c_proj Linear layer is created correctly
    if "self.c_proj = Linear(intermediate_size, embed_dim, bias=True)" in source:
        print("✅ self.c_proj Linear layer is created correctly")
    else:
        print("❌ self.c_proj Linear layer is not created correctly")
        print("   Hint: Use Linear(intermediate_size, embed_dim, bias=True)")

    # Test 7: Check if c_fc is applied to hidden_states
    if "self.c_fc(hidden_states)" in source:
        print("✅ self.c_fc is applied to hidden_states")
    else:
        print("❌ self.c_fc is not applied to hidden_states")
        print("   Hint: Apply self.c_fc to hidden_states in the __call__ method")

    # Test 8: Check if F.gelu is used with correct parameters
    if "F.gelu" in source:
        if 'approximate="tanh"' in source or "approximate='tanh'" in source:
            print("✅ F.gelu is used with approximate='tanh'")
        else:
            print("❌ F.gelu is not used with correct parameters")
            print('   Hint: Use F.gelu(hidden_states, approximate="tanh")')
    else:
        print("❌ F.gelu is not used")
        print("   Hint: Use F.gelu() for the activation function")

    # Test 9: Check if c_proj is applied to hidden_states
    if "self.c_proj(hidden_states)" in source:
        print("✅ self.c_proj is applied to hidden_states")
    else:
        print("❌ self.c_proj is not applied to hidden_states")
        print("   Hint: Apply self.c_proj to hidden_states after the activation")

    # Test 10: Check that None values are replaced
    lines = source.split("\n")
    none_assignments = [
        line
        for line in lines
        if (
            "self.c_fc = None" in line
            or "self.c_proj = None" in line
            or "hidden_states = None" in line
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

    # Test 11: Try to instantiate the GPT2MLP class
    try:
        from solutions.solution_01 import GPT2Config

        config = GPT2Config()
        mlp = mlp_module.GPT2MLP(intermediate_size=3072, config=config)
        print("✅ GPT2MLP class can be instantiated")

        # Check if c_fc and c_proj are initialized
        if hasattr(mlp, "c_fc") and mlp.c_fc is not None:
            print("✅ GPT2MLP.c_fc is initialized")
        else:
            print("❌ GPT2MLP.c_fc is not initialized")

        if hasattr(mlp, "c_proj") and mlp.c_proj is not None:
            print("✅ GPT2MLP.c_proj is initialized")
        else:
            print("❌ GPT2MLP.c_proj is not initialized")

    except Exception as e:
        print(f"❌ GPT2MLP class instantiation failed: {e}")
        print("   This usually means some TODO items are not completed")

    # Test 12: Try to run the forward pass (if imports are available)
    try:
        from max.experimental.tensor import Tensor

        # Create a dummy input tensor
        # Shape: (batch_size=1, sequence_length=4, embedding_dim=768)
        hidden_states = Tensor.ones([1, 4, 768])

        result = mlp(hidden_states)
        print("✅ GPT2MLP forward pass executes without errors")

        # Check output shape
        if hasattr(result, "shape"):
            expected_shape = (1, 4, 768)
            actual_shape = tuple(result.shape)
            if actual_shape == expected_shape:
                print(f"✅ Output shape is correct: {actual_shape}")
            else:
                print(
                    f"❌ Output shape is incorrect: expected {expected_shape}, got {actual_shape}"
                )

    except ImportError:
        print(
            "❌  Cannot test forward pass execution (MAX not available in this environment)"
        )
    except AttributeError as e:
        print(f"❌ Forward pass execution failed: {e}")
        print("   This usually means some TODO items are not completed")
    except Exception as e:
        print(f"❌ Forward pass execution failed with error: {e}")

    # Final summary
    print("\n" + "=" * 60)
    if all(
        [
            "from max.experimental import functional as F" in source,
            "from max.experimental.tensor import Tensor" in source,
            "from max.nn.module_v3 import" in source,
            "self.c_fc = Linear(embed_dim, intermediate_size, bias=True)" in source,
            "self.c_proj = Linear(intermediate_size, embed_dim, bias=True)" in source,
            "self.c_fc(hidden_states)" in source,
            "F.gelu" in source,
            ('approximate="tanh"' in source or "approximate='tanh'" in source),
            "self.c_proj(hidden_states)" in source,
            not none_assignments,
        ]
    ):
        print("🎉 All checks passed! Your implementation matches the solution.")
        print("=" * 60)
    else:
        print("⚠️  Some checks failed. Review the hints above and try again.")
        print("=" * 60)


if __name__ == "__main__":
    test_step_04()

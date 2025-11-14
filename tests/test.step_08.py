"""Tests for Step 08: Residual Connections and Layer Normalization"""

import ast
from pathlib import Path


def test_step_08():
    """Comprehensive validation for Step 08 implementation."""

    results = []
    step_file = Path("steps/step_08.py")

    # Read source
    if not step_file.exists():
        print(f"❌ File not found: {step_file}")
        return

    source = step_file.read_text()
    tree = ast.parse(source)

    # Phase 1: Import checks
    has_functional = False
    has_tensor = False
    has_dimlike = False
    has_module = False

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module == "max.experimental":
                for alias in node.names:
                    if alias.name == "functional" or (
                        alias.asname and alias.asname == "F"
                    ):
                        has_functional = True
            if node.module == "max.experimental.tensor":
                for alias in node.names:
                    if alias.name == "Tensor":
                        has_tensor = True
            if node.module == "max.graph":
                for alias in node.names:
                    if alias.name == "DimLike":
                        has_dimlike = True
            if node.module == "max.nn.module_v3":
                for alias in node.names:
                    if alias.name == "Module":
                        has_module = True

    if has_functional:
        results.append("✅ functional is correctly imported from max.experimental")
    else:
        results.append("❌ functional is not imported from max.experimental")
        results.append("   Hint: Add 'from max.experimental import functional as F'")

    if has_tensor:
        results.append("✅ Tensor is correctly imported from max.experimental.tensor")
    else:
        results.append("❌ Tensor is not imported from max.experimental.tensor")
        results.append("   Hint: Add 'from max.experimental.tensor import Tensor'")

    if has_dimlike:
        results.append("✅ DimLike is correctly imported from max.graph")
    else:
        results.append("❌ DimLike is not imported from max.graph")
        results.append("   Hint: Add 'from max.graph import DimLike'")

    if has_module:
        results.append("✅ Module is correctly imported from max.nn.module_v3")
    else:
        results.append("❌ Module is not imported from max.nn.module_v3")
        results.append("   Hint: Add 'from max.nn.module_v3 import Module'")

    # Phase 2: Structure checks
    try:
        from steps.step_08 import LayerNorm, ResidualBlock, apply_residual_connection

        results.append("✅ LayerNorm class exists")
        results.append("✅ ResidualBlock class exists")
        results.append("✅ apply_residual_connection function exists")
    except ImportError as e:
        if "LayerNorm" in str(e):
            results.append("❌ LayerNorm class not found in step_08 module")
            results.append("   Hint: Create class LayerNorm(Module)")
        if "ResidualBlock" in str(e):
            results.append("❌ ResidualBlock class not found in step_08 module")
            results.append("   Hint: Create class ResidualBlock(Module)")
        if "apply_residual_connection" in str(e):
            results.append("❌ apply_residual_connection function not found")
            results.append("   Hint: Define apply_residual_connection function")
        print("\n".join(results))
        return

    # Check inheritance
    from max.nn.module_v3 import Module

    if issubclass(LayerNorm, Module):
        results.append("✅ LayerNorm inherits from Module")
    else:
        results.append("❌ LayerNorm must inherit from Module")

    if issubclass(ResidualBlock, Module):
        results.append("✅ ResidualBlock inherits from Module")
    else:
        results.append("❌ ResidualBlock must inherit from Module")

    # Phase 3: Implementation checks
    # Check LayerNorm
    if "Tensor.ones" in source:
        results.append("✅ LayerNorm uses Tensor.ones for weight")
    else:
        results.append("❌ LayerNorm should use Tensor.ones for weight")
        results.append("   Hint: self.weight = Tensor.ones([dim])")

    if "Tensor.zeros" in source:
        results.append("✅ LayerNorm uses Tensor.zeros for bias")
    else:
        results.append("❌ LayerNorm should use Tensor.zeros for bias")
        results.append("   Hint: self.bias = Tensor.zeros([dim])")

    if "F.layer_norm" in source:
        results.append("✅ LayerNorm uses F.layer_norm")
    else:
        results.append("❌ LayerNorm should use F.layer_norm")
        results.append(
            "   Hint: return F.layer_norm(x, gamma=self.weight, beta=self.bias, epsilon=self.eps)"
        )

    if "gamma=self.weight" in source or "gamma = self.weight" in source:
        results.append("✅ LayerNorm passes weight as gamma parameter")
    else:
        results.append("❌ LayerNorm should pass self.weight as gamma")
        results.append("   Hint: gamma=self.weight in F.layer_norm call")

    if "beta=self.bias" in source or "beta = self.bias" in source:
        results.append("✅ LayerNorm passes bias as beta parameter")
    else:
        results.append("❌ LayerNorm should pass self.bias as beta")
        results.append("   Hint: beta=self.bias in F.layer_norm call")

    # Check ResidualBlock
    if "self.ln = LayerNorm" in source or (
        "self.ln =" in source
        and "None" not in source.split("self.ln =")[1].split("\n")[0]
    ):
        results.append("✅ ResidualBlock creates LayerNorm instance")
    else:
        results.append("❌ ResidualBlock should create LayerNorm instance")
        results.append("   Hint: self.ln = LayerNorm(dim, eps=eps)")

    # Check residual connections (addition)
    if (
        source.count(" + ") >= 2 or source.count("+") >= 4
    ):  # At least in ResidualBlock and apply_residual_connection
        results.append("✅ Residual connections use addition operator")
    else:
        results.append("❌ Residual connections should use + operator")
        results.append("   Hint: return x + sublayer_output")

    # Phase 4: Placeholder detection
    none_lines = [
        line.strip()
        for line in source.split("\n")
        if ("= None" in line or "return None" in line)
        and not line.strip().startswith("#")
        and "def " not in line
        and "Optional" not in line
    ]
    if none_lines:
        results.append("❌ Found placeholder 'None' values that need to be replaced:")
        for line in none_lines[:5]:
            results.append(f"   {line}")
        results.append(
            "   Hint: Replace all 'None' values with the actual implementation"
        )
    else:
        results.append("✅ All placeholder 'None' values have been replaced")

    # Phase 5: Functional tests
    try:
        from max.driver import CPU
        from max.dtype import DType
        from max.experimental.tensor import Tensor
        import numpy as np

        # Test LayerNorm
        dim = 768
        ln = LayerNorm(dim, eps=1e-5)
        results.append("✅ LayerNorm class can be instantiated")

        # Check attributes
        if hasattr(ln, "weight"):
            results.append("✅ LayerNorm.weight is initialized")
        else:
            results.append("❌ LayerNorm.weight attribute not found")

        if hasattr(ln, "bias"):
            results.append("✅ LayerNorm.bias is initialized")
        else:
            results.append("❌ LayerNorm.bias attribute not found")

        # Test forward pass
        batch_size = 2
        seq_length = 8
        test_input = Tensor.randn(
            batch_size, seq_length, dim, dtype=DType.float32, device=CPU()
        )

        output = ln(test_input)
        results.append("✅ LayerNorm forward pass executes without errors")

        # Check output shape
        expected_shape = (batch_size, seq_length, dim)
        if output.shape == expected_shape:
            results.append(f"✅ LayerNorm output shape is correct: {expected_shape}")
        else:
            results.append(
                f"❌ LayerNorm output shape incorrect: expected {expected_shape}, got {output.shape}"
            )

        # Check normalization properties (mean ≈ 0, std ≈ 1)
        output_np = np.from_dlpack(output.to(CPU()))
        mean = output_np.mean(axis=-1)
        std = output_np.std(axis=-1)

        if np.allclose(mean, 0, atol=1e-5):
            results.append("✅ LayerNorm output has mean ≈ 0 (normalized)")
        else:
            results.append(
                f"⚠️ LayerNorm output mean is {mean.mean():.6f} (expected ≈ 0)"
            )

        if np.allclose(std, 1, atol=1e-4):
            results.append("✅ LayerNorm output has std ≈ 1 (normalized)")
        else:
            results.append(f"⚠️ LayerNorm output std is {std.mean():.6f} (expected ≈ 1)")

        # Test ResidualBlock
        rb = ResidualBlock(dim, eps=1e-5)
        results.append("✅ ResidualBlock class can be instantiated")

        if hasattr(rb, "ln"):
            results.append("✅ ResidualBlock.ln is initialized")
        else:
            results.append("❌ ResidualBlock.ln attribute not found")

        # Test residual connection
        test_residual = Tensor.randn(
            batch_size, seq_length, dim, dtype=DType.float32, device=CPU()
        )
        test_sublayer = Tensor.randn(
            batch_size, seq_length, dim, dtype=DType.float32, device=CPU()
        )

        residual_output = rb(test_residual, test_sublayer)
        results.append("✅ ResidualBlock forward pass executes without errors")

        # Check that output equals input + sublayer_output
        expected_output_np = np.from_dlpack(test_residual.to(CPU())) + np.from_dlpack(
            test_sublayer.to(CPU())
        )
        residual_output_np = np.from_dlpack(residual_output.to(CPU()))

        if np.allclose(residual_output_np, expected_output_np, atol=1e-5):
            results.append("✅ ResidualBlock correctly adds input + sublayer_output")
        else:
            results.append(
                "❌ ResidualBlock output incorrect (should be input + sublayer_output)"
            )

        # Test apply_residual_connection function
        func_output = apply_residual_connection(test_residual, test_sublayer)
        results.append("✅ apply_residual_connection executes without errors")

        func_output_np = np.from_dlpack(func_output.to(CPU()))
        if np.allclose(func_output_np, expected_output_np, atol=1e-5):
            results.append("✅ apply_residual_connection correctly adds tensors")
        else:
            results.append("❌ apply_residual_connection output incorrect")

    except Exception as e:
        results.append(f"❌ Functional test failed: {e}")
        import traceback

        tb = traceback.format_exc()
        error_lines = [line for line in tb.split("\n") if line.strip()]
        if error_lines:
            results.append(f"   {error_lines[-1]}")

    # Print all results
    print(
        "Running tests for Step 08: Residual Connections and Layer Normalization...\n"
    )
    print("Results:")
    print("\n".join(results))

    # Summary
    failed = any(r.startswith("❌") for r in results)
    if not failed:
        print("\n" + "=" * 60)
        print("🎉 All checks passed! Your implementation is complete.")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("⚠️ Some checks failed. Review the hints above and try again.")
        print("=" * 60)


if __name__ == "__main__":
    test_step_10()

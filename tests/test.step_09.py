"""Tests for Step 09: Transformer Block"""

import ast
from pathlib import Path


def test_step_09():
    """Comprehensive validation for Step 09 implementation."""

    results = []
    step_file = Path("steps/step_09.py")

    # Read source
    if not step_file.exists():
        print(f"❌ File not found: {step_file}")
        return

    source = step_file.read_text()
    tree = ast.parse(source)

    # Phase 1: Import checks
    has_module = False
    has_config = False
    has_mlp = False
    has_attention = False
    has_layernorm = False

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module == "max.nn.module_v3":
                for alias in node.names:
                    if alias.name == "Module":
                        has_module = True
            if node.module == "solutions.solution_01":
                for alias in node.names:
                    if alias.name == "GPT2Config":
                        has_config = True
            if node.module == "solutions.solution_04":
                for alias in node.names:
                    if alias.name == "GPT2MLP":
                        has_mlp = True
            if node.module == "solutions.solution_09":
                for alias in node.names:
                    if alias.name == "GPT2MultiHeadAttention":
                        has_attention = True
            if node.module == "solutions.solution_10":
                for alias in node.names:
                    if alias.name == "LayerNorm":
                        has_layernorm = True

    if has_module:
        results.append("✅ Module is correctly imported from max.nn.module_v3")
    else:
        results.append("❌ Module is not imported from max.nn.module_v3")
        results.append("   Hint: Add 'from max.nn.module_v3 import Module'")

    if has_config:
        results.append("✅ GPT2Config is correctly imported from solutions.solution_01")
    else:
        results.append("❌ GPT2Config is not imported")
        results.append("   Hint: Add 'from solutions.solution_01 import GPT2Config'")

    if has_mlp:
        results.append("✅ GPT2MLP is correctly imported from solutions.solution_04")
    else:
        results.append("❌ GPT2MLP is not imported")
        results.append("   Hint: Add 'from solutions.solution_04 import GPT2MLP'")

    if has_attention:
        results.append(
            "✅ GPT2MultiHeadAttention is correctly imported from solutions.solution_09"
        )
    else:
        results.append("❌ GPT2MultiHeadAttention is not imported")
        results.append(
            "   Hint: Add 'from solutions.solution_09 import GPT2MultiHeadAttention'"
        )

    if has_layernorm:
        results.append("✅ LayerNorm is correctly imported from solutions.solution_10")
    else:
        results.append("❌ LayerNorm is not imported")
        results.append("   Hint: Add 'from solutions.solution_10 import LayerNorm'")

    # Phase 2: Structure checks
    try:
        from steps.step_09 import GPT2Block

        results.append("✅ GPT2Block class exists")
    except ImportError:
        results.append("❌ GPT2Block class not found in step_09 module")
        results.append("   Hint: Create class GPT2Block(Module)")
        print("\n".join(results))
        return

    # Check inheritance
    from max.nn.module_v3 import Module

    if issubclass(GPT2Block, Module):
        results.append("✅ GPT2Block inherits from Module")
    else:
        results.append("❌ GPT2Block must inherit from Module")

    # Phase 3: Implementation checks
    if "self.ln_1 = LayerNorm" in source or (
        "self.ln_1 =" in source
        and "None" not in source.split("self.ln_1 =")[1].split("\n")[0]
    ):
        results.append("✅ self.ln_1 is created correctly")
    else:
        results.append("❌ self.ln_1 layer norm is not created correctly")
        results.append(
            "   Hint: self.ln_1 = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)"
        )

    if "self.attn = GPT2MultiHeadAttention" in source or (
        "self.attn =" in source
        and "None" not in source.split("self.attn =")[1].split("\n")[0]
    ):
        results.append("✅ self.attn is created correctly")
    else:
        results.append("❌ self.attn is not created correctly")
        results.append("   Hint: self.attn = GPT2MultiHeadAttention(config)")

    if "self.ln_2 = LayerNorm" in source or (
        "self.ln_2 =" in source
        and "None" not in source.split("self.ln_2 =")[1].split("\n")[0]
    ):
        results.append("✅ self.ln_2 is created correctly")
    else:
        results.append("❌ self.ln_2 layer norm is not created correctly")
        results.append(
            "   Hint: self.ln_2 = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)"
        )

    if "self.mlp = GPT2MLP" in source or (
        "self.mlp =" in source
        and "None" not in source.split("self.mlp =")[1].split("\n")[0]
    ):
        results.append("✅ self.mlp is created correctly")
    else:
        results.append("❌ self.mlp is not created correctly")
        results.append("   Hint: self.mlp = GPT2MLP(inner_dim, config)")

    # Check forward pass structure
    if "self.ln_1(hidden_states)" in source.replace(" ", ""):
        results.append("✅ Forward pass calls ln_1")
    else:
        results.append("❌ Forward pass should call self.ln_1(hidden_states)")

    if "self.attn(" in source:
        results.append("✅ Forward pass calls attn")
    else:
        results.append("❌ Forward pass should call self.attn")

    if "self.ln_2(" in source:
        results.append("✅ Forward pass calls ln_2")
    else:
        results.append("❌ Forward pass should call self.ln_2")

    if "self.mlp(" in source:
        results.append("✅ Forward pass calls mlp")
    else:
        results.append("❌ Forward pass should call self.mlp")

    # Check residual connections
    residual_count = source.count("residual =")
    if residual_count >= 2:
        results.append("✅ Forward pass uses residual connections (at least 2)")
    else:
        results.append(
            "❌ Forward pass should store residual twice (before attn and mlp)"
        )
        results.append(
            f"   Found {residual_count} residual assignments, need at least 2"
        )

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
        from solutions.solution_01 import GPT2Config
        import numpy as np

        config = GPT2Config()
        block = GPT2Block(config)
        results.append("✅ GPT2Block class can be instantiated")

        # Check attributes
        if hasattr(block, "ln_1"):
            results.append("✅ GPT2Block.ln_1 is initialized")
        else:
            results.append("❌ GPT2Block.ln_1 attribute not found")

        if hasattr(block, "attn"):
            results.append("✅ GPT2Block.attn is initialized")
        else:
            results.append("❌ GPT2Block.attn attribute not found")

        if hasattr(block, "ln_2"):
            results.append("✅ GPT2Block.ln_2 is initialized")
        else:
            results.append("❌ GPT2Block.ln_2 attribute not found")

        if hasattr(block, "mlp"):
            results.append("✅ GPT2Block.mlp is initialized")
        else:
            results.append("❌ GPT2Block.mlp attribute not found")

        # Test forward pass
        batch_size = 2
        seq_length = 8
        test_input = Tensor.randn(
            batch_size, seq_length, config.n_embd, dtype=DType.float32, device=CPU()
        )

        output = block(test_input)
        results.append("✅ GPT2Block forward pass executes without errors")

        # Check output shape
        expected_shape = (batch_size, seq_length, config.n_embd)
        if output.shape == expected_shape:
            results.append(f"✅ Output shape is correct: {expected_shape}")
        else:
            results.append(
                f"❌ Output shape incorrect: expected {expected_shape}, got {output.shape}"
            )

        # Check output contains non-zero values
        output_np = np.from_dlpack(output.to(CPU()))
        if not np.allclose(output_np, 0):
            results.append("✅ Output contains non-zero values")
        else:
            results.append("❌ Output is all zeros")

    except Exception as e:
        results.append(f"❌ Functional test failed: {e}")
        import traceback

        tb = traceback.format_exc()
        error_lines = [line for line in tb.split("\n") if line.strip()]
        if error_lines:
            results.append(f"   {error_lines[-1]}")

    # Print all results
    print("Running tests for Step 09: Transformer Block...\n")
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
    test_step_09()

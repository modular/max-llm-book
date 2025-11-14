"""Tests for Step 06: Position Embeddings"""

import ast
import inspect
from pathlib import Path


def test_step_06():
    """Comprehensive validation for Step 06 implementation."""

    results = []
    step_file = Path("steps/step_06.py")

    # Read source
    if not step_file.exists():
        print(f"❌ File not found: {step_file}")
        return

    source = step_file.read_text()
    tree = ast.parse(source)

    # Phase 1: Import checks
    has_tensor = False
    has_embedding = False
    has_module = False

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module == "max.experimental.tensor":
                for alias in node.names:
                    if alias.name == "Tensor":
                        has_tensor = True
            if node.module == "max.nn.module_v3":
                for alias in node.names:
                    if alias.name == "Embedding":
                        has_embedding = True
                    if alias.name == "Module":
                        has_module = True

    if has_tensor:
        results.append("✅ Tensor is correctly imported from max.experimental.tensor")
    else:
        results.append("❌ Tensor is not imported from max.experimental.tensor")
        results.append("   Hint: Add 'from max.experimental.tensor import Tensor'")

    if has_embedding:
        results.append("✅ Embedding is correctly imported from max.nn.module_v3")
    else:
        results.append("❌ Embedding is not imported from max.nn.module_v3")
        results.append("   Hint: Add 'from max.nn.module_v3 import Embedding, Module'")

    if has_module:
        results.append("✅ Module is correctly imported from max.nn.module_v3")
    else:
        results.append("❌ Module is not imported from max.nn.module_v3")
        results.append("   Hint: Add 'from max.nn.module_v3 import Embedding, Module'")

    # Phase 2: Structure checks
    try:
        from steps.step_06 import GPT2PositionEmbeddings

        results.append("✅ GPT2PositionEmbeddings class exists")
    except ImportError:
        results.append("❌ GPT2PositionEmbeddings class not found in step_06 module")
        results.append("   Hint: Create class GPT2PositionEmbeddings(Module)")
        print("\n".join(results))
        return

    # Check inheritance
    from max.nn.module_v3 import Module

    if issubclass(GPT2PositionEmbeddings, Module):
        results.append("✅ GPT2PositionEmbeddings inherits from Module")
    else:
        results.append("❌ GPT2PositionEmbeddings must inherit from Module")

    # Phase 3: Implementation checks
    if "self.wpe = Embedding" in source or (
        "self.wpe =" in source
        and "None" not in source.split("self.wpe =")[1].split("\n")[0]
    ):
        results.append("✅ self.wpe embedding layer is created correctly")
    else:
        results.append("❌ self.wpe embedding layer is not created correctly")
        results.append("   Hint: Use Embedding(config.n_positions, dim=config.n_embd)")

    # Check if n_positions is used
    if "config.n_positions" in source:
        results.append("✅ config.n_positions is used correctly")
    else:
        results.append("❌ config.n_positions not found")
        results.append("   Hint: First parameter should be config.n_positions")

    # Check if n_embd is used
    if "config.n_embd" in source:
        results.append("✅ config.n_embd is used correctly")
    else:
        results.append("❌ config.n_embd not found")
        results.append("   Hint: Use dim=config.n_embd for the embedding dimension")

    # Check forward pass
    if "self.wpe(position_ids)" in source.replace(" ", ""):
        results.append("✅ self.wpe is called with position_ids in __call__ method")
    else:
        results.append("❌ self.wpe is not called with position_ids")
        results.append("   Hint: Return self.wpe(position_ids) in the __call__ method")

    # Phase 4: Placeholder detection
    none_lines = [
        line.strip()
        for line in source.split("\n")
        if "= None" in line
        and not line.strip().startswith("#")
        and "def " not in line
        and ":" not in line.split("=")[0]
    ]
    if none_lines:
        results.append("❌ Found placeholder 'None' values that need to be replaced:")
        for line in none_lines[:3]:
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

        config = GPT2Config()
        pos_embeddings = GPT2PositionEmbeddings(config)
        results.append("✅ GPT2PositionEmbeddings class can be instantiated")

        # Check wpe attribute exists
        if hasattr(pos_embeddings, "wpe"):
            results.append("✅ GPT2PositionEmbeddings.wpe is initialized")
        else:
            results.append("❌ GPT2PositionEmbeddings.wpe attribute not found")

        # Test forward pass with position indices
        # Create position indices for a sequence
        seq_length = 8
        test_positions = Tensor.arange(seq_length, dtype=DType.int64, device=CPU())

        output = pos_embeddings(test_positions)
        results.append("✅ GPT2PositionEmbeddings forward pass executes without errors")

        # Check output shape
        expected_shape = (seq_length, config.n_embd)
        if output.shape == expected_shape:
            results.append(f"✅ Output shape is correct: {expected_shape}")
        else:
            results.append(
                f"❌ Output shape is incorrect: expected {expected_shape}, got {output.shape}"
            )

        # Verify output is not all zeros (embeddings should have values)
        import numpy as np

        output_np = np.from_dlpack(output.to(CPU()))
        if not np.allclose(output_np, 0):
            results.append("✅ Output contains non-zero embedding values")
        else:
            results.append("❌ Output is all zeros - embeddings may not be initialized")

        # Test that different positions give different embeddings
        if seq_length > 1:
            first_pos = output_np[0]
            second_pos = output_np[1]
            if not np.allclose(first_pos, second_pos):
                results.append("✅ Different positions produce different embeddings")
            else:
                results.append("⚠️ Warning: Position 0 and 1 have identical embeddings")

    except Exception as e:
        results.append(f"❌ Functional test failed: {e}")
        import traceback

        results.append(f"   {traceback.format_exc().split('Error:')[-1].strip()}")

    # Print all results
    print("Running tests for Step 06: Position Embeddings...\n")
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
    test_step_06()

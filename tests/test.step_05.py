"""Tests for Step 05: Token Embeddings"""

import ast
import inspect
from pathlib import Path


def test_step_05():
    """Comprehensive validation for Step 05 implementation."""

    results = []
    step_file = Path("steps/step_05.py")

    # Read source
    if not step_file.exists():
        print(f"❌ File not found: {step_file}")
        return

    source = step_file.read_text()
    tree = ast.parse(source)

    # Phase 1: Import checks
    has_embedding = False
    has_module = False

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module == "max.nn.module_v3":
                for alias in node.names:
                    if alias.name == "Embedding":
                        has_embedding = True
                    if alias.name == "Module":
                        has_module = True

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
        from steps.step_05 import GPT2Embeddings

        results.append("✅ GPT2Embeddings class exists")
    except ImportError:
        results.append("❌ GPT2Embeddings class not found in step_05 module")
        results.append("   Hint: Create class GPT2Embeddings(Module)")
        print("\n".join(results))
        return

    # Check inheritance
    if issubclass(GPT2Embeddings, Module):
        results.append("✅ GPT2Embeddings inherits from Module")
    else:
        results.append("❌ GPT2Embeddings must inherit from Module")

    # Phase 3: Implementation checks
    if "self.wte = Embedding" in source or (
        "self.wte =" in source
        and "None" not in source.split("self.wte =")[1].split("\n")[0]
    ):
        results.append("✅ self.wte embedding layer is created correctly")
    else:
        results.append("❌ self.wte embedding layer is not created correctly")
        results.append("   Hint: Use Embedding(config.vocab_size, dim=config.n_embd)")

    # Check if vocab_size is used
    if "config.vocab_size" in source:
        results.append("✅ config.vocab_size is used correctly")
    else:
        results.append("❌ config.vocab_size not found")
        results.append("   Hint: First parameter should be config.vocab_size")

    # Check if n_embd is used
    if "config.n_embd" in source:
        results.append("✅ config.n_embd is used correctly")
    else:
        results.append("❌ config.n_embd not found")
        results.append("   Hint: Use dim=config.n_embd for the embedding dimension")

    # Check forward pass
    if "self.wte(input_ids)" in source.replace(" ", ""):
        results.append("✅ self.wte is called with input_ids in __call__ method")
    else:
        results.append("❌ self.wte is not called with input_ids")
        results.append("   Hint: Return self.wte(input_ids) in the __call__ method")

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
        embeddings = GPT2Embeddings(config)
        results.append("✅ GPT2Embeddings class can be instantiated")

        # Check wte attribute exists
        if hasattr(embeddings, "wte"):
            results.append("✅ GPT2Embeddings.wte is initialized")
        else:
            results.append("❌ GPT2Embeddings.wte attribute not found")

        # Test forward pass with sample token IDs
        # Create a small batch of token IDs
        batch_size, seq_length = 2, 4
        test_input = Tensor.constant(
            [[1, 2, 3, 4], [5, 6, 7, 8]], dtype=DType.int64, device=CPU()
        )

        output = embeddings(test_input)
        results.append("✅ GPT2Embeddings forward pass executes without errors")

        # Check output shape
        expected_shape = (batch_size, seq_length, config.n_embd)
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

    except Exception as e:
        results.append(f"❌ Functional test failed: {e}")
        import traceback

        results.append(f"   {traceback.format_exc().split('Error:')[-1].strip()}")

    # Print all results
    print("Running tests for Step 05: Token Embeddings...\n")
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
    test_step_05()

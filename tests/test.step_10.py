"""Tests for Step 10: Stacking Transformer Blocks"""

import ast
from pathlib import Path


def test_step_10():
    """Comprehensive validation for Step 10 implementation."""

    results = []
    step_file = Path("steps/step_10.py")

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
    has_sequential = False
    has_config = False
    has_layernorm = False
    has_block = False

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
                    if alias.name == "Sequential":
                        has_sequential = True
            if node.module == "solutions.solution_01":
                for alias in node.names:
                    if alias.name == "GPT2Config":
                        has_config = True
            if node.module == "solutions.solution_10":
                for alias in node.names:
                    if alias.name == "LayerNorm":
                        has_layernorm = True
            if node.module == "solutions.solution_11":
                for alias in node.names:
                    if alias.name == "GPT2Block":
                        has_block = True

    if has_tensor:
        results.append("✅ Tensor is correctly imported from max.experimental.tensor")
    else:
        results.append("❌ Tensor is not imported from max.experimental.tensor")
        results.append("   Hint: Add 'from max.experimental.tensor import Tensor'")

    if has_embedding:
        results.append("✅ Embedding is correctly imported from max.nn.module_v3")
    else:
        results.append("❌ Embedding is not imported from max.nn.module_v3")
        results.append(
            "   Hint: Add 'from max.nn.module_v3 import Embedding, Module, Sequential'"
        )

    if has_module:
        results.append("✅ Module is correctly imported from max.nn.module_v3")
    else:
        results.append("❌ Module is not imported from max.nn.module_v3")
        results.append(
            "   Hint: Add 'from max.nn.module_v3 import Embedding, Module, Sequential'"
        )

    if has_sequential:
        results.append("✅ Sequential is correctly imported from max.nn.module_v3")
    else:
        results.append("❌ Sequential is not imported from max.nn.module_v3")
        results.append(
            "   Hint: Add 'from max.nn.module_v3 import Embedding, Module, Sequential'"
        )

    if has_config:
        results.append("✅ GPT2Config is correctly imported")
    else:
        results.append("❌ GPT2Config is not imported")
        results.append("   Hint: Add 'from solutions.solution_01 import GPT2Config'")

    if has_layernorm:
        results.append("✅ LayerNorm is correctly imported")
    else:
        results.append("❌ LayerNorm is not imported")
        results.append("   Hint: Add 'from solutions.solution_10 import LayerNorm'")

    if has_block:
        results.append("✅ GPT2Block is correctly imported")
    else:
        results.append("❌ GPT2Block is not imported")
        results.append("   Hint: Add 'from solutions.solution_11 import GPT2Block'")

    # Phase 2: Structure checks
    try:
        from steps.step_10 import GPT2Model

        results.append("✅ GPT2Model class exists")
    except ImportError:
        results.append("❌ GPT2Model class not found in step_10 module")
        results.append("   Hint: Create class GPT2Model(Module)")
        print("\n".join(results))
        return

    # Check inheritance
    from max.nn.module_v3 import Module

    if issubclass(GPT2Model, Module):
        results.append("✅ GPT2Model inherits from Module")
    else:
        results.append("❌ GPT2Model must inherit from Module")

    # Phase 3: Implementation checks
    if "self.wte = Embedding" in source or (
        "self.wte =" in source
        and "None" not in source.split("self.wte =")[1].split("\n")[0]
    ):
        results.append("✅ self.wte token embeddings created correctly")
    else:
        results.append("❌ self.wte is not created correctly")
        results.append(
            "   Hint: self.wte = Embedding(config.vocab_size, dim=config.n_embd)"
        )

    if "self.wpe = Embedding" in source or (
        "self.wpe =" in source
        and "None" not in source.split("self.wpe =")[1].split("\n")[0]
    ):
        results.append("✅ self.wpe position embeddings created correctly")
    else:
        results.append("❌ self.wpe is not created correctly")
        results.append(
            "   Hint: self.wpe = Embedding(config.n_positions, dim=config.n_embd)"
        )

    if "self.h = Sequential" in source or (
        "self.h =" in source
        and "Sequential" in source
        and "None" not in source.split("self.h =")[1].split("\n")[0]
    ):
        results.append("✅ self.h transformer blocks stack created correctly")
    else:
        results.append("❌ self.h is not created correctly")
        results.append(
            "   Hint: self.h = Sequential(*(GPT2Block(config) for _ in range(config.n_layer)))"
        )

    if "self.ln_f = LayerNorm" in source or (
        "self.ln_f =" in source
        and "None" not in source.split("self.ln_f =")[1].split("\n")[0]
    ):
        results.append("✅ self.ln_f final layer norm created correctly")
    else:
        results.append("❌ self.ln_f is not created correctly")
        results.append(
            "   Hint: self.ln_f = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)"
        )

    # Check forward pass
    if "input_ids.shape" in source:
        results.append("✅ Forward pass extracts shape from input_ids")
    else:
        results.append(
            "❌ Should extract batch_size and seq_length from input_ids.shape"
        )

    if "self.wte(" in source and "__call__" in source:
        results.append("✅ Forward pass calls wte for token embeddings")
    else:
        results.append("❌ Forward pass should call self.wte(input_ids)")

    if "Tensor.arange" in source:
        results.append("✅ Forward pass uses Tensor.arange for positions")
    else:
        results.append(
            "❌ Forward pass should use Tensor.arange to create position indices"
        )

    if "self.wpe(" in source and "__call__" in source:
        results.append("✅ Forward pass calls wpe for position embeddings")
    else:
        results.append("❌ Forward pass should call self.wpe on position indices")

    if "tok_embeds + pos_embeds" in source.replace(" ", "") or "+ pos_embeds" in source:
        results.append("✅ Forward pass combines token and position embeddings")
    else:
        results.append("❌ Should add tok_embeds + pos_embeds")

    if "self.h(" in source and "__call__" in source:
        results.append("✅ Forward pass applies transformer blocks (self.h)")
    else:
        results.append("❌ Forward pass should call self.h(x)")

    if "self.ln_f(" in source and "__call__" in source:
        results.append("✅ Forward pass applies final layer norm")
    else:
        results.append("❌ Forward pass should call self.ln_f(x)")

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
        model = GPT2Model(config)
        results.append("✅ GPT2Model class can be instantiated")

        # Check attributes
        if hasattr(model, "wte"):
            results.append("✅ GPT2Model.wte is initialized")
        else:
            results.append("❌ GPT2Model.wte attribute not found")

        if hasattr(model, "wpe"):
            results.append("✅ GPT2Model.wpe is initialized")
        else:
            results.append("❌ GPT2Model.wpe attribute not found")

        if hasattr(model, "h"):
            results.append("✅ GPT2Model.h is initialized")
        else:
            results.append("❌ GPT2Model.h attribute not found")

        if hasattr(model, "ln_f"):
            results.append("✅ GPT2Model.ln_f is initialized")
        else:
            results.append("❌ GPT2Model.ln_f attribute not found")

        # Test forward pass
        batch_size = 2
        seq_length = 8
        # Create random token IDs
        test_input = Tensor.randint(
            0,
            config.vocab_size,
            batch_size,
            seq_length,
            dtype=DType.int64,
            device=CPU(),
        )

        output = model(test_input)
        results.append("✅ GPT2Model forward pass executes without errors")

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
    print("Running tests for Step 10: Stacking Transformer Blocks...\n")
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

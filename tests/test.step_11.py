"""Tests for Step 11: Language Model Head"""

import ast
from pathlib import Path


def test_step_11():
    """Comprehensive validation for Step 11 implementation."""

    results = []
    step_file = Path("steps/step_11.py")

    # Read source
    if not step_file.exists():
        print(f"❌ File not found: {step_file}")
        return

    source = step_file.read_text()

    # Phase 1: Import checks
    has_linear = "from max.nn.module_v3 import" in source and "Linear" in source
    has_module = "from max.nn.module_v3 import" in source and "Module" in source
    has_config = "from solutions.solution_01 import GPT2Config" in source
    has_model = "from solutions.solution_12 import GPT2Model" in source

    if has_linear:
        results.append("✅ Linear is correctly imported")
    else:
        results.append("❌ Linear is not imported")
        results.append("   Hint: Add 'from max.nn.module_v3 import Linear, Module'")

    if has_module:
        results.append("✅ Module is correctly imported")
    else:
        results.append("❌ Module is not imported")
        results.append("   Hint: Add 'from max.nn.module_v3 import Linear, Module'")

    if has_config:
        results.append("✅ GPT2Config is correctly imported")
    else:
        results.append("❌ GPT2Config is not imported")
        results.append("   Hint: Add 'from solutions.solution_01 import GPT2Config'")

    if has_model:
        results.append("✅ GPT2Model is correctly imported")
    else:
        results.append("❌ GPT2Model is not imported")
        results.append("   Hint: Add 'from solutions.solution_12 import GPT2Model'")

    # Phase 2: Structure checks
    try:
        from steps.step_11 import MaxGPT2LMHeadModel

        results.append("✅ MaxGPT2LMHeadModel class exists")
    except ImportError:
        results.append("❌ MaxGPT2LMHeadModel class not found")
        results.append("   Hint: Create class MaxGPT2LMHeadModel(Module)")
        print("\n".join(results))
        return

    # Phase 3: Implementation checks
    if "self.transformer = GPT2Model" in source or (
        "self.transformer =" in source
        and "None" not in source.split("self.transformer =")[1].split("\n")[0]
    ):
        results.append("✅ self.transformer is created correctly")
    else:
        results.append("❌ self.transformer is not created correctly")
        results.append("   Hint: self.transformer = GPT2Model(config)")

    if "self.lm_head = Linear" in source or (
        "self.lm_head =" in source
        and "None" not in source.split("self.lm_head =")[1].split("\n")[0]
    ):
        results.append("✅ self.lm_head is created correctly")
    else:
        results.append("❌ self.lm_head is not created correctly")
        results.append(
            "   Hint: self.lm_head = Linear(config.n_embd, config.vocab_size, bias=False)"
        )

    if "bias=False" in source or "bias = False" in source:
        results.append("✅ lm_head uses bias=False")
    else:
        results.append("❌ lm_head should use bias=False")
        results.append("   Hint: Add bias=False to Linear layer")

    if "self.transformer(" in source and "__call__" in source:
        results.append("✅ Forward pass calls transformer")
    else:
        results.append("❌ Forward pass should call self.transformer(input_ids)")

    if "self.lm_head(" in source and "__call__" in source:
        results.append("✅ Forward pass calls lm_head")
    else:
        results.append("❌ Forward pass should call self.lm_head")

    # Phase 4: Placeholder detection
    none_lines = [
        line.strip()
        for line in source.split("\n")
        if ("= None" in line or "return None" in line)
        and not line.strip().startswith("#")
        and "def " not in line
    ]
    if none_lines:
        results.append("❌ Found placeholder 'None' values that need to be replaced:")
        for line in none_lines[:3]:
            results.append(f"   {line}")
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
        model = MaxGPT2LMHeadModel(config)
        results.append("✅ MaxGPT2LMHeadModel can be instantiated")

        if hasattr(model, "transformer"):
            results.append("✅ MaxGPT2LMHeadModel.transformer is initialized")
        else:
            results.append("❌ transformer attribute not found")

        if hasattr(model, "lm_head"):
            results.append("✅ MaxGPT2LMHeadModel.lm_head is initialized")
        else:
            results.append("❌ lm_head attribute not found")

        # Test forward pass
        batch_size = 2
        seq_length = 8
        test_input = Tensor.randint(
            0,
            config.vocab_size,
            batch_size,
            seq_length,
            dtype=DType.int64,
            device=CPU(),
        )

        output = model(test_input)
        results.append("✅ Forward pass executes without errors")

        # Check output shape
        expected_shape = (batch_size, seq_length, config.vocab_size)
        if output.shape == expected_shape:
            results.append(f"✅ Output shape is correct: {expected_shape}")
        else:
            results.append(
                f"❌ Output shape incorrect: expected {expected_shape}, got {output.shape}"
            )

        output_np = np.from_dlpack(output.to(CPU()))
        if not np.allclose(output_np, 0):
            results.append("✅ Output contains non-zero values")
        else:
            results.append("❌ Output is all zeros")

    except Exception as e:
        results.append(f"❌ Functional test failed: {e}")

    # Print all results
    print("Running tests for Step 11: Language Model Head...\n")
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
    test_step_11()

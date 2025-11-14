"""Tests for Step 12: Text Generation"""

import ast
from pathlib import Path


def test_step_12():
    """Comprehensive validation for Step 12 implementation."""

    results = []
    step_file = Path("steps/step_12.py")

    # Read source
    if not step_file.exists():
        print(f"❌ File not found: {step_file}")
        return

    source = step_file.read_text()

    # Phase 1: Import checks
    has_numpy = "import numpy" in source
    has_cpu = "from max.driver import" in source and "CPU" in source
    has_dtype = "from max.dtype import" in source and "DType" in source
    has_functional = "from max.experimental import" in source and "functional" in source
    has_tensor = "from max.experimental.tensor import" in source and "Tensor" in source

    if has_numpy:
        results.append("✅ numpy is correctly imported")
    else:
        results.append("❌ numpy is not imported")
        results.append("   Hint: Add 'import numpy as np'")

    if has_cpu:
        results.append("✅ CPU is correctly imported")
    else:
        results.append("❌ CPU is not imported")
        results.append("   Hint: Add 'from max.driver import CPU'")

    if has_dtype:
        results.append("✅ DType is correctly imported")
    else:
        results.append("❌ DType is not imported")
        results.append("   Hint: Add 'from max.dtype import DType'")

    if has_functional:
        results.append("✅ functional is correctly imported")
    else:
        results.append("❌ functional is not imported")
        results.append("   Hint: Add 'from max.experimental import functional as F'")

    if has_tensor:
        results.append("✅ Tensor is correctly imported")
    else:
        results.append("❌ Tensor is not imported")
        results.append("   Hint: Add 'from max.experimental.tensor import Tensor'")

    # Phase 2: Structure checks
    try:
        from steps.step_12 import generate_next_token, generate_tokens

        results.append("✅ generate_next_token function exists")
        results.append("✅ generate_tokens function exists")
    except ImportError as e:
        if "generate_next_token" in str(e):
            results.append("❌ generate_next_token function not found")
            results.append("   Hint: Define generate_next_token function")
        if "generate_tokens" in str(e):
            results.append("❌ generate_tokens function not found")
            results.append("   Hint: Define generate_tokens function")
        print("\n".join(results))
        return

    # Phase 3: Implementation checks
    if "model(input_ids)" in source.replace(" ", ""):
        results.append("✅ Calls model to get logits")
    else:
        results.append("❌ Should call model(input_ids) to get logits")

    if "logits[0, -1, :]" in source.replace(
        " ", ""
    ) or "logits[0,-1,:]" in source.replace(" ", ""):
        results.append("✅ Extracts last position logits correctly")
    else:
        results.append("❌ Should extract logits[0, -1, :] for next token")

    if "Tensor.constant" in source and "temperature" in source:
        results.append("✅ Creates temperature tensor")
    else:
        results.append("❌ Should create temperature tensor with Tensor.constant")

    if "F.softmax" in source:
        results.append("✅ Uses F.softmax for probabilities")
    else:
        results.append("❌ Should use F.softmax to convert logits to probabilities")

    if "np.random.choice" in source:
        results.append("✅ Uses np.random.choice for sampling")
    else:
        results.append("❌ Should use np.random.choice for sampling")

    if "F.argmax" in source:
        results.append("✅ Uses F.argmax for greedy decoding")
    else:
        results.append("❌ Should use F.argmax for greedy decoding")

    if "range(max_new_tokens)" in source.replace(" ", ""):
        results.append("✅ Has generation loop")
    else:
        results.append("❌ Should have loop: for _ in range(max_new_tokens)")

    if "F.concat" in source:
        results.append("✅ Uses F.concat to append tokens")
    else:
        results.append("❌ Should use F.concat to append new tokens")

    # Phase 4: Placeholder detection
    none_lines = [
        line.strip()
        for line in source.split("\n")
        if ("return None" in line) and not line.strip().startswith("#")
    ]
    if none_lines:
        results.append("❌ Found placeholder 'return None' values:")
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
        from solutions.solution_13 import MaxGPT2LMHeadModel

        config = GPT2Config()
        model = MaxGPT2LMHeadModel(config)

        # Test generate_next_token
        batch_size = 1
        seq_length = 5
        test_input = Tensor.randint(
            0,
            config.vocab_size,
            batch_size,
            seq_length,
            dtype=DType.int64,
            device=CPU(),
        )

        next_token = generate_next_token(
            model, test_input, temperature=1.0, do_sample=True
        )
        results.append("✅ generate_next_token executes without errors (sampling)")

        next_token_greedy = generate_next_token(
            model, test_input, temperature=1.0, do_sample=False
        )
        results.append("✅ generate_next_token executes without errors (greedy)")

        # Test generate_tokens
        generated = generate_tokens(
            model, test_input, max_new_tokens=3, temperature=1.0, do_sample=True
        )
        results.append("✅ generate_tokens executes without errors")

        expected_length = seq_length + 3
        if generated.shape[1] == expected_length:
            results.append(
                f"✅ Generated sequence has correct length: {expected_length}"
            )
        else:
            results.append(
                f"❌ Expected length {expected_length}, got {generated.shape[1]}"
            )

        # Check that generated tokens are different from input
        import numpy as np

        generated_np = np.from_dlpack(generated.to(CPU()))
        test_input_np = np.from_dlpack(test_input.to(CPU()))

        if generated_np.shape[1] > test_input_np.shape[1]:
            results.append("✅ New tokens were generated")
        else:
            results.append("❌ No new tokens generated")

    except Exception as e:
        results.append(f"❌ Functional test failed: {e}")
        import traceback

        tb = traceback.format_exc()
        error_lines = [line for line in tb.split("\n") if line.strip()]
        if error_lines:
            results.append(f"   {error_lines[-1]}")

    # Print all results
    print("Running tests for Step 12: Text Generation...\n")
    print("Results:")
    print("\n".join(results))

    # Summary
    failed = any(r.startswith("❌") for r in results)
    if not failed:
        print("\n" + "=" * 60)
        print("🎉 All checks passed! Your implementation is complete.")
        print("=" * 60)
        print("\n🎊 Congratulations! You've completed all 12 steps!")
        print("You've built a complete GPT-2 model from scratch in MAX!")
    else:
        print("\n" + "=" * 60)
        print("⚠️ Some checks failed. Review the hints above and try again.")
        print("=" * 60)


if __name__ == "__main__":
    test_step_12()

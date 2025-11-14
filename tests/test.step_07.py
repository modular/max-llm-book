"""Tests for Step 07: Multi-head Attention"""

import ast
from pathlib import Path


def test_step_07():
    """Comprehensive validation for Step 07 implementation."""

    results = []
    step_file = Path("steps/step_07.py")

    # Read source
    if not step_file.exists():
        print(f"❌ File not found: {step_file}")
        return

    source = step_file.read_text()
    tree = ast.parse(source)

    # Phase 1: Import checks
    has_math = False
    has_linear = False
    has_module = False
    has_functional = False

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "math":
                    has_math = True
        if isinstance(node, ast.ImportFrom):
            if node.module == "max.nn.module_v3":
                for alias in node.names:
                    if alias.name == "Linear":
                        has_linear = True
                    if alias.name == "Module":
                        has_module = True
            if node.module == "max.experimental":
                for alias in node.names:
                    if alias.name == "functional" or (
                        alias.asname and alias.asname == "F"
                    ):
                        has_functional = True

    if has_math:
        results.append("✅ math is correctly imported")
    else:
        results.append("❌ math is not imported")
        results.append("   Hint: Add 'import math'")

    if has_linear:
        results.append("✅ Linear is correctly imported from max.nn.module_v3")
    else:
        results.append("❌ Linear is not imported from max.nn.module_v3")
        results.append("   Hint: Add 'from max.nn.module_v3 import Linear, Module'")

    if has_module:
        results.append("✅ Module is correctly imported from max.nn.module_v3")
    else:
        results.append("❌ Module is not imported from max.nn.module_v3")
        results.append("   Hint: Add 'from max.nn.module_v3 import Linear, Module'")

    if has_functional:
        results.append("✅ functional is correctly imported from max.experimental")
    else:
        results.append("❌ functional is not imported from max.experimental")
        results.append("   Hint: Add 'from max.experimental import functional as F'")

    # Phase 2: Structure checks
    try:
        from steps.step_07 import GPT2MultiHeadAttention, causal_mask

        results.append("✅ GPT2MultiHeadAttention class exists")
        results.append("✅ causal_mask function exists")
    except ImportError as e:
        if "GPT2MultiHeadAttention" in str(e):
            results.append(
                "❌ GPT2MultiHeadAttention class not found in step_07 module"
            )
            results.append("   Hint: Create class GPT2MultiHeadAttention(Module)")
        if "causal_mask" in str(e):
            results.append("❌ causal_mask function not found")
            results.append("   Hint: Copy causal_mask from solution_02.py")
        print("\n".join(results))
        return

    # Check inheritance
    from max.nn.module_v3 import Module

    if issubclass(GPT2MultiHeadAttention, Module):
        results.append("✅ GPT2MultiHeadAttention inherits from Module")
    else:
        results.append("❌ GPT2MultiHeadAttention must inherit from Module")

    # Phase 3: Implementation checks
    # Check layer creation
    if "self.c_attn = Linear" in source or (
        "self.c_attn =" in source
        and "None" not in source.split("self.c_attn =")[1].split("\n")[0]
    ):
        results.append("✅ self.c_attn linear layer is created correctly")
    else:
        results.append("❌ self.c_attn linear layer is not created correctly")
        results.append(
            "   Hint: Use Linear(self.embed_dim, 3 * self.embed_dim, bias=True)"
        )

    if "self.c_proj = Linear" in source or (
        "self.c_proj =" in source
        and "None" not in source.split("self.c_proj =")[1].split("\n")[0]
    ):
        results.append("✅ self.c_proj linear layer is created correctly")
    else:
        results.append("❌ self.c_proj linear layer is not created correctly")
        results.append("   Hint: Use Linear(self.embed_dim, self.embed_dim, bias=True)")

    # Check _split_heads
    if "tensor.reshape" in source and "_split_heads" in source:
        results.append("✅ _split_heads uses tensor.reshape")
    else:
        results.append("❌ _split_heads should use tensor.reshape")
        results.append("   Hint: tensor = tensor.reshape(new_shape)")

    if "transpose(-3, -2)" in source.replace(
        " ", ""
    ) or "transpose(-3,-2)" in source.replace(" ", ""):
        results.append("✅ _split_heads uses transpose(-3, -2)")
    else:
        results.append("❌ _split_heads should use transpose(-3, -2)")
        results.append("   Hint: return tensor.transpose(-3, -2)")

    # Check _merge_heads
    merge_heads_source = (
        source[source.find("def _merge_heads") : source.find("def _attn")]
        if "def _attn" in source
        else source[source.find("def _merge_heads") :]
    )
    if "transpose" in merge_heads_source and "reshape" in merge_heads_source:
        results.append("✅ _merge_heads uses transpose and reshape")
    else:
        results.append("❌ _merge_heads should use transpose and reshape")
        results.append(
            "   Hint: tensor.transpose(-3, -2) then tensor.reshape(new_shape)"
        )

    # Check _attn
    if "query @ key.transpose(-1, -2)" in source.replace(" ", ""):
        results.append("✅ _attn computes Q @ K^T")
    else:
        results.append("❌ _attn should compute query @ key.transpose(-1, -2)")
        results.append("   Hint: Copy attention computation from Step 08")

    if source.count("F.softmax") > 0:
        results.append("✅ _attn uses F.softmax")
    else:
        results.append("❌ _attn should use F.softmax")
        results.append("   Hint: Copy attention computation from Step 08")

    # Check forward pass
    if "self.c_attn(hidden_states)" in source.replace(" ", ""):
        results.append("✅ Forward pass projects with c_attn")
    else:
        results.append("❌ Forward pass should call self.c_attn(hidden_states)")
        results.append("   Hint: qkv = self.c_attn(hidden_states)")

    if "F.split" in source and "__call__" in source:
        results.append("✅ Forward pass splits Q, K, V")
    else:
        results.append("❌ Forward pass should use F.split to separate Q, K, V")
        results.append("   Hint: query, key, value = F.split(qkv, ...)")

    if "self._split_heads" in source:
        results.append("✅ Forward pass calls _split_heads")
    else:
        results.append("❌ Forward pass should call self._split_heads")
        results.append(
            "   Hint: query = self._split_heads(query, self.num_heads, self.head_dim)"
        )

    if "self._attn" in source:
        results.append("✅ Forward pass calls _attn")
    else:
        results.append("❌ Forward pass should call self._attn")
        results.append("   Hint: attn_output = self._attn(query, key, value)")

    if "self._merge_heads" in source:
        results.append("✅ Forward pass calls _merge_heads")
    else:
        results.append("❌ Forward pass should call self._merge_heads")
        results.append("   Hint: attn_output = self._merge_heads(attn_output, ...)")

    if "self.c_proj" in source and "__call__" in source:
        results.append("✅ Forward pass uses c_proj for output projection")
    else:
        results.append("❌ Forward pass should call self.c_proj")
        results.append("   Hint: attn_output = self.c_proj(attn_output)")

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
        mha = GPT2MultiHeadAttention(config)
        results.append("✅ GPT2MultiHeadAttention class can be instantiated")

        # Check attributes
        if hasattr(mha, "c_attn"):
            results.append("✅ GPT2MultiHeadAttention.c_attn is initialized")
        else:
            results.append("❌ GPT2MultiHeadAttention.c_attn attribute not found")

        if hasattr(mha, "c_proj"):
            results.append("✅ GPT2MultiHeadAttention.c_proj is initialized")
        else:
            results.append("❌ GPT2MultiHeadAttention.c_proj attribute not found")

        # Test forward pass
        batch_size = 2
        seq_length = 8
        test_input = Tensor.randn(
            batch_size, seq_length, config.n_embd, dtype=DType.float32, device=CPU()
        )

        output = mha(test_input)
        results.append("✅ GPT2MultiHeadAttention forward pass executes without errors")

        # Check output shape
        expected_shape = (batch_size, seq_length, config.n_embd)
        if output.shape == expected_shape:
            results.append(f"✅ Output shape is correct: {expected_shape}")
        else:
            results.append(
                f"❌ Output shape is incorrect: expected {expected_shape}, got {output.shape}"
            )

        # Check output contains non-zero values
        output_np = np.from_dlpack(output.to(CPU()))
        if not np.allclose(output_np, 0):
            results.append("✅ Output contains non-zero values")
        else:
            results.append("❌ Output is all zeros")

        # Test _split_heads
        test_tensor = Tensor.randn(
            batch_size, seq_length, config.n_embd, dtype=DType.float32, device=CPU()
        )
        split_output = mha._split_heads(test_tensor, config.n_head, mha.head_dim)
        expected_split_shape = (batch_size, config.n_head, seq_length, mha.head_dim)
        if split_output.shape == expected_split_shape:
            results.append(
                f"✅ _split_heads output shape is correct: {expected_split_shape}"
            )
        else:
            results.append(
                f"❌ _split_heads output shape incorrect: expected {expected_split_shape}, got {split_output.shape}"
            )

        # Test _merge_heads
        merge_output = mha._merge_heads(split_output, config.n_head, mha.head_dim)
        expected_merge_shape = (batch_size, seq_length, config.n_embd)
        if merge_output.shape == expected_merge_shape:
            results.append(
                f"✅ _merge_heads output shape is correct: {expected_merge_shape}"
            )
        else:
            results.append(
                f"❌ _merge_heads output shape incorrect: expected {expected_merge_shape}, got {merge_output.shape}"
            )

    except Exception as e:
        results.append(f"❌ Functional test failed: {e}")
        import traceback

        tb = traceback.format_exc()
        error_lines = [line for line in tb.split("\n") if line.strip()]
        if error_lines:
            results.append(f"   {error_lines[-1]}")

    # Print all results
    print("Running tests for Step 07: Multi-head Attention...\n")
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
    test_step_07()

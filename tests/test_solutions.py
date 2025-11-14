"""Test all solutions to ensure they work correctly with current MAX APIs.

These tests verify that:
1. All solution modules can be imported without errors
2. Classes and functions are defined with correct signatures
3. Solutions integrate correctly with current MAX API

Note: We avoid full forward passes with random data since those require
actual model weights and specific MAX tensor creation patterns.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_solution_01():
    """Test solution 01: Model Configuration"""
    from solutions.solution_01 import GPT2Config

    config = GPT2Config()
    assert config.vocab_size == 50257
    assert config.n_positions == 1024
    assert config.n_embd == 768
    assert config.n_layer == 12
    assert config.n_head == 12
    assert config.n_inner == 3072
    assert config.layer_norm_epsilon == 1e-05
    print("✅ Solution 01 passed")


def test_solution_02():
    """Test solution 02: Causal Masking"""
    from solutions.solution_02 import causal_mask
    import inspect

    # Verify function exists and has correct signature
    assert callable(causal_mask)
    sig = inspect.signature(causal_mask)
    assert "sequence_length" in sig.parameters
    assert "num_tokens" in sig.parameters
    assert "dtype" in sig.parameters
    assert "device" in sig.parameters
    print("✅ Solution 02 passed")


def test_solution_03():
    """Test solution 03: Layer Normalization"""
    from solutions.solution_03 import LayerNorm
    from max.nn.module_v3 import Module

    # Verify LayerNorm class exists and is a Module
    ln = LayerNorm(768)
    assert isinstance(ln, Module)
    assert hasattr(ln, "weight")
    assert hasattr(ln, "bias")
    assert hasattr(ln, "eps")
    print("✅ Solution 03 passed")


def test_solution_04():
    """Test solution 04: Feed-Forward Network (MLP)"""
    from solutions.solution_01 import GPT2Config
    from solutions.solution_04 import GPT2MLP
    from max.nn.module_v3 import Module

    # Verify GPT2MLP class exists and has correct structure
    config = GPT2Config()
    mlp = GPT2MLP(config.n_inner, config)
    assert isinstance(mlp, Module)
    assert hasattr(mlp, "c_fc")
    assert hasattr(mlp, "c_proj")
    print("✅ Solution 04 passed")


def test_solution_05():
    """Test solution 05: Token Embeddings"""
    from solutions.solution_01 import GPT2Config
    from solutions.solution_05 import GPT2Embeddings
    from max.nn.module_v3 import Module

    # Verify GPT2Embeddings class exists and has correct structure
    config = GPT2Config()
    embeddings = GPT2Embeddings(config)
    assert isinstance(embeddings, Module)
    assert hasattr(embeddings, "wte")
    print("✅ Solution 05 passed")


def test_solution_06():
    """Test solution 06: Position Embeddings"""
    from solutions.solution_01 import GPT2Config
    from solutions.solution_06 import GPT2PositionEmbeddings
    from max.nn.module_v3 import Module

    # Verify GPT2PositionEmbeddings class exists and has correct structure
    config = GPT2Config()
    pos_embeddings = GPT2PositionEmbeddings(config)
    assert isinstance(pos_embeddings, Module)
    assert hasattr(pos_embeddings, "wpe")
    print("✅ Solution 06 passed")


def test_solution_07():
    """Test solution 07: Multi-head Attention"""
    from solutions.solution_01 import GPT2Config
    from solutions.solution_07 import GPT2MultiHeadAttention, causal_mask
    from max.nn.module_v3 import Module
    import inspect

    # Verify GPT2MultiHeadAttention class exists and has correct structure
    config = GPT2Config()
    attn = GPT2MultiHeadAttention(config)
    assert isinstance(attn, Module)
    assert hasattr(attn, "c_attn")
    assert hasattr(attn, "c_proj")
    assert hasattr(attn, "num_heads")
    assert hasattr(attn, "head_dim")

    # Verify causal_mask function exists
    assert callable(causal_mask)
    print("✅ Solution 07 passed")


def test_solution_08():
    """Test solution 08: Residual Connections and Layer Normalization"""
    from solutions.solution_08 import (
        LayerNorm,
        ResidualBlock,
        apply_residual_connection,
    )
    from max.nn.module_v3 import Module

    # Verify LayerNorm class
    ln = LayerNorm(768)
    assert isinstance(ln, Module)
    assert hasattr(ln, "weight")
    assert hasattr(ln, "bias")

    # Verify ResidualBlock class
    rb = ResidualBlock(768)
    assert isinstance(rb, Module)
    assert hasattr(rb, "ln")

    # Verify apply_residual_connection function
    assert callable(apply_residual_connection)
    print("✅ Solution 08 passed")


def test_solution_09():
    """Test solution 09: Transformer Block"""
    from solutions.solution_01 import GPT2Config
    from solutions.solution_09 import GPT2Block
    from max.nn.module_v3 import Module

    # Verify GPT2Block class exists and has correct structure
    config = GPT2Config()
    block = GPT2Block(config)
    assert isinstance(block, Module)
    assert hasattr(block, "ln_1")
    assert hasattr(block, "attn")
    assert hasattr(block, "ln_2")
    assert hasattr(block, "mlp")
    print("✅ Solution 09 passed")


def test_solution_10():
    """Test solution 10: Stacking Transformer Blocks"""
    from solutions.solution_01 import GPT2Config
    from solutions.solution_10 import GPT2Model
    from max.nn.module_v3 import Module

    # Verify GPT2Model class exists and has correct structure
    config = GPT2Config()
    model = GPT2Model(config)
    assert isinstance(model, Module)
    assert hasattr(model, "wte")
    assert hasattr(model, "wpe")
    assert hasattr(model, "h")
    assert hasattr(model, "ln_f")
    # Check that we have the right number of blocks
    assert len(model.h) == config.n_layer
    print("✅ Solution 10 passed")


def test_solution_11():
    """Test solution 11: Language Model Head"""
    from solutions.solution_01 import GPT2Config
    from solutions.solution_11 import MaxGPT2LMHeadModel
    from max.nn.module_v3 import Module

    # Verify MaxGPT2LMHeadModel class exists and has correct structure
    config = GPT2Config()
    model = MaxGPT2LMHeadModel(config)
    assert isinstance(model, Module)
    assert hasattr(model, "transformer")
    assert hasattr(model, "lm_head")
    assert hasattr(model, "config")
    print("✅ Solution 11 passed")


def test_solution_12():
    """Test solution 12: Text Generation"""
    from solutions.solution_12 import generate_next_token, generate_tokens
    import inspect

    # Verify generation functions exist and have correct signatures
    assert callable(generate_next_token)
    assert callable(generate_tokens)

    # Check signatures
    gen_next_sig = inspect.signature(generate_next_token)
    assert "model" in gen_next_sig.parameters
    assert "input_ids" in gen_next_sig.parameters
    assert "temperature" in gen_next_sig.parameters

    gen_tokens_sig = inspect.signature(generate_tokens)
    assert "model" in gen_tokens_sig.parameters
    assert "input_ids" in gen_tokens_sig.parameters
    assert "max_new_tokens" in gen_tokens_sig.parameters
    print("✅ Solution 12 passed")


if __name__ == "__main__":
    tests = [
        test_solution_01,
        test_solution_02,
        test_solution_03,
        test_solution_04,
        test_solution_05,
        test_solution_06,
        test_solution_07,
        test_solution_08,
        test_solution_09,
        test_solution_10,
        test_solution_11,
        test_solution_12,
    ]

    print("Testing all solutions...\n")
    failed = []

    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")
            failed.append(test.__name__)

    print("\n" + "=" * 60)
    if not failed:
        print("🎉 All solution tests passed!")
    else:
        print(f"⚠️  {len(failed)} test(s) failed:")
        for name in failed:
            print(f"  - {name}")
        sys.exit(1)
    print("=" * 60)

# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
from unittest.mock import Mock, patch

import numpy as np
from gpt2 import (
    GPT2MLP,
    GPT2Block,
    GPT2Config,
    GPT2GreedyHead,
    GPT2MultiHeadAttention,
    GPT2SamplingHead,
    LayerNorm,
    MaxGPT2LMHeadModel,
    MaxGPT2Model,
    causal_mask,
    decode_tokens,
    encode_text,
)
from max.driver import CPU
from max.dtype import DType
from max.experimental.tensor import Tensor


class TestGPT2Config:
    """Test GPT2Config dataclass."""

    def test_default_values(self) -> None:
        """Test that GPT2Config has correct default values."""
        config = GPT2Config()
        assert config.vocab_size == 50257
        assert config.n_positions == 1024
        assert config.n_embd == 768
        assert config.n_layer == 12
        assert config.n_head == 12
        assert config.n_inner is None
        assert config.layer_norm_epsilon == 1e-5

    def test_custom_values(self) -> None:
        """Test that GPT2Config can be instantiated with custom values."""
        config = GPT2Config()
        config.n_embd = 512
        config.n_layer = 6
        assert config.n_embd == 512
        assert config.n_layer == 6


class TestGPT2MLP:
    """Test GPT2MLP module."""

    def test_initialization(self) -> None:
        """Test that GPT2MLP initializes correctly."""
        config = GPT2Config()
        mlp = GPT2MLP(intermediate_size=3072, config=config)
        assert mlp.c_fc is not None
        assert mlp.c_proj is not None

    def test_forward_pass_shape(self) -> None:
        """Test that GPT2MLP produces correct output shape."""
        config = GPT2Config()
        mlp = GPT2MLP(intermediate_size=3072, config=config)

        # Create input tensor [batch=2, seq=10, embd=768]
        hidden_states = Tensor.ones([2, 10, config.n_embd], dtype=DType.float32)
        output = mlp(hidden_states)

        # Output should have same shape as input
        assert output.shape == hidden_states.shape


class TestCausalMask:
    """Test causal_mask function."""

    def test_causal_mask_shape(self) -> None:
        """Test that causal mask has correct shape."""
        seq_len = 5
        num_tokens = 0
        mask = causal_mask(
            seq_len, num_tokens, dtype=DType.float32, device=CPU()
        )
        # Compare as lists of ints since MAX returns Dim objects
        assert [int(d) for d in mask.shape] == [seq_len, seq_len]

    def test_causal_mask_values(self) -> None:
        """Test that causal mask has correct values (upper triangle is -inf)."""
        seq_len = 4
        num_tokens = 0
        mask = causal_mask(
            seq_len, num_tokens, dtype=DType.float32, device=CPU()
        )

        # Convert to numpy for inspection
        mask_np = np.from_dlpack(mask)

        # Lower triangle and diagonal should be 0
        for i in range(seq_len):
            for j in range(i + 1):
                assert mask_np[i, j] == 0.0

        # Upper triangle should be -inf
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                assert mask_np[i, j] == float("-inf")


class TestGPT2MultiHeadAttention:
    """Test GPT2MultiHeadAttention module."""

    def test_initialization(self) -> None:
        """Test that GPT2MultiHeadAttention initializes correctly."""
        config = GPT2Config()
        attn = GPT2MultiHeadAttention(config)
        assert attn.embed_dim == config.n_embd
        assert attn.num_heads == config.n_head
        assert attn.head_dim == config.n_embd // config.n_head
        assert attn.c_attn is not None
        assert attn.c_proj is not None

    def test_split_heads_shape(self) -> None:
        """Test that _split_heads produces correct shape."""
        config = GPT2Config()
        attn = GPT2MultiHeadAttention(config)

        batch_size, seq_len = 2, 10
        tensor = Tensor.ones(
            [batch_size, seq_len, config.n_embd], dtype=DType.float32
        )
        split = attn._split_heads(tensor, config.n_head, attn.head_dim)

        # Should be [batch, n_head, seq_len, head_dim]
        assert [int(d) for d in split.shape] == [
            batch_size,
            config.n_head,
            seq_len,
            attn.head_dim,
        ]

    def test_merge_heads_shape(self) -> None:
        """Test that _merge_heads produces correct shape."""
        config = GPT2Config()
        attn = GPT2MultiHeadAttention(config)

        batch_size, seq_len = 2, 10
        tensor = Tensor.ones(
            [batch_size, config.n_head, seq_len, attn.head_dim],
            dtype=DType.float32,
        )
        merged = attn._merge_heads(tensor, config.n_head, attn.head_dim)

        # Should be [batch, seq_len, n_embd]
        assert [int(d) for d in merged.shape] == [
            batch_size,
            seq_len,
            config.n_embd,
        ]

    def test_forward_pass_shape(self) -> None:
        """Test that attention forward pass produces correct output shape."""
        config = GPT2Config()
        attn = GPT2MultiHeadAttention(config)

        hidden_states = Tensor.ones([2, 10, config.n_embd], dtype=DType.float32)
        output = attn(hidden_states)

        assert output.shape == hidden_states.shape


class TestLayerNorm:
    """Test LayerNorm module."""

    def test_initialization(self) -> None:
        """Test that LayerNorm initializes correctly."""
        dim = 768
        ln = LayerNorm(dim, eps=1e-5)
        assert ln.eps == 1e-5
        assert [int(d) for d in ln.weight.shape] == [dim]
        assert [int(d) for d in ln.bias.shape] == [dim]

    def test_forward_pass_shape(self) -> None:
        """Test that LayerNorm produces correct output shape."""
        dim = 768
        ln = LayerNorm(dim)

        x = Tensor.ones([2, 10, dim])
        output = ln(x)

        assert output.shape == x.shape


class TestGPT2Block:
    """Test GPT2Block module."""

    def test_initialization(self) -> None:
        """Test that GPT2Block initializes correctly."""
        config = GPT2Config()
        block = GPT2Block(config)
        assert block.ln_1 is not None
        assert block.attn is not None
        assert block.ln_2 is not None
        assert block.mlp is not None

    def test_initialization_with_custom_inner_dim(self) -> None:
        """Test that GPT2Block uses custom inner_dim when provided."""
        config = GPT2Config()
        config.n_inner = 2048
        block = GPT2Block(config)
        assert block.mlp is not None

    def test_forward_pass_shape(self) -> None:
        """Test that GPT2Block produces correct output shape."""
        config = GPT2Config()
        block = GPT2Block(config)

        hidden_states = Tensor.ones([2, 10, config.n_embd])
        output = block(hidden_states)

        assert output.shape == hidden_states.shape


class TestMaxGPT2Model:
    """Test MaxGPT2Model module."""

    def test_initialization(self) -> None:
        """Test that MaxGPT2Model initializes correctly."""
        config = GPT2Config()
        model = MaxGPT2Model(config)
        assert model.wte is not None
        assert model.wpe is not None
        assert model.h is not None
        assert model.ln_f is not None

    def test_forward_pass_shape(self) -> None:
        """Test that MaxGPT2Model produces correct output shape."""
        config = GPT2Config()
        model = MaxGPT2Model(config)

        batch_size, seq_len = 2, 10
        input_ids = Tensor.zeros([batch_size, seq_len], dtype=DType.int64)
        output = model(input_ids)

        # Output should be [batch, seq_len, n_embd]
        assert [int(d) for d in output.shape] == [
            batch_size,
            seq_len,
            config.n_embd,
        ]


class TestMaxGPT2LMHeadModel:
    """Test MaxGPT2LMHeadModel module."""

    def test_initialization(self) -> None:
        """Test that MaxGPT2LMHeadModel initializes correctly."""
        config = GPT2Config()
        model = MaxGPT2LMHeadModel(config)
        assert model.config == config
        assert model.transformer is not None
        assert model.lm_head is not None

    def test_forward_pass_shape(self) -> None:
        """Test that MaxGPT2LMHeadModel produces correct output shape."""
        config = GPT2Config()
        model = MaxGPT2LMHeadModel(config)

        batch_size, seq_len = 2, 10
        input_ids = Tensor.zeros([batch_size, seq_len], dtype=DType.int64)
        output = model(input_ids)

        # Output should be [batch, seq_len, vocab_size]
        assert [int(d) for d in output.shape] == [
            batch_size,
            seq_len,
            config.vocab_size,
        ]


class TestTokenizationFunctions:
    """Test tokenization and decoding functions."""

    @patch("main.GPT2Tokenizer")
    def test_encode_text(self, mock_tokenizer_class: Mock) -> None:
        """Test encode_text returns list[int] from tokenizer."""
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [15496, 995]  # "Hello world"

        result = encode_text("Hello world", mock_tokenizer, max_length=128)

        mock_tokenizer.encode.assert_called_once_with(
            "Hello world", max_length=128, truncation=True
        )
        assert isinstance(result, list)
        assert result == [15496, 995]

    @patch("main.GPT2Tokenizer")
    def test_decode_tokens(self, mock_tokenizer_class: Mock) -> None:
        """Test decode_tokens accepts list[int] and returns decoded string."""
        mock_tokenizer = Mock()
        mock_tokenizer.decode.return_value = "Hello world"

        result = decode_tokens([15496, 995], mock_tokenizer)

        mock_tokenizer.decode.assert_called_once_with(
            [15496, 995], skip_special_tokens=True
        )
        assert result == "Hello world"


class TestSamplingHeads:
    """Test GPT2SamplingHead and GPT2GreedyHead module structure."""

    def test_sampling_head_initialization(self) -> None:
        """Test that GPT2SamplingHead initializes correctly."""
        config = GPT2Config()
        lm_head = MaxGPT2LMHeadModel(config)
        head = GPT2SamplingHead(lm_head)
        assert head.lm_head is lm_head

    def test_greedy_head_initialization(self) -> None:
        """Test that GPT2GreedyHead initializes correctly."""
        config = GPT2Config()
        lm_head = MaxGPT2LMHeadModel(config)
        head = GPT2GreedyHead(lm_head)
        assert head.lm_head is lm_head


class TestModelDimensions:
    """Test that model dimensions are consistent throughout."""

    def test_head_dimensions(self) -> None:
        """Test that head dimensions divide evenly."""
        config = GPT2Config()
        assert config.n_embd % config.n_head == 0
        head_dim = config.n_embd // config.n_head
        assert head_dim == 64

    def test_mlp_inner_dimension(self) -> None:
        """Test default MLP inner dimension is 4x embedding."""
        config = GPT2Config()
        expected_inner = 4 * config.n_embd
        assert expected_inner == 3072

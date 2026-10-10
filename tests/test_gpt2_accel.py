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
"""GPU-targeted variant of test_gpt2.py.

Runs the same checks as ``test_gpt2.py`` but on an ``Accelerator()``
device. Forward-pass tests build the module on CPU, transfer it to the
accelerator via ``Module.to(...)``, run both, and compare element-wise.
On Metal, the backend has a known bug where ``seq_len > 3`` produces
incorrect codegen; the affected forward-pass tests are therefore marked
``xfail(raises=AssertionError, strict=True)``, so the suite reports
green on Apple Silicon while flagging an XPASS when the underlying bug
is fixed.

Pure shape/mask tests do not exercise the buggy kernel path and run
straight on Metal without an xfail marker.
"""

import os
from collections.abc import Callable

import max.experimental.functional as F
import numpy as np
import pytest
from gpt2_arch.gpt2 import (
    GPT2MLP,
    GPT2Block,
    GPT2Config,
    GPT2MultiHeadAttention,
    LayerNorm,
    MaxGPT2LMHeadModel,
    MaxGPT2Model,
    causal_mask,
)
from max._interpreter_ops import GC_FAMILIES, adopted_from_manifest
from max.driver import CPU, Accelerator, accelerator_api, accelerator_count
from max.dtype import DType
from max.experimental.nn import Module
from max.experimental.tensor import Tensor, default_device, default_dtype


def _probe_accelerator() -> Accelerator:
    """Return an Accelerator after confirming it can compile a Mojo kernel.

    Mirrors the probe in ``notebooks/tutorial.ipynb``: exercise the kernel
    path with a small GELU so we fail fast if the runtime detects an
    accelerator but cannot actually use it.
    """
    gpu = Accelerator()
    with default_device(gpu), default_dtype(DType.float32):
        probe = F.gelu(Tensor.ones([1, 4]), approximate="tanh")
        _ = np.from_dlpack(probe.to(CPU()))
    return gpu


# Probe at import time but report any failure via pytestmark so pytest
# still collects tests and exits cleanly (a module-level pytest.skip causes
# pytest to exit with code 5 "no tests collected", which bazel treats as a
# failure).
GPU: Accelerator | None = None
_SKIP_REASON: str | None = None

if accelerator_count() == 0:
    _SKIP_REASON = "No accelerator detected; test_gpt2_accel requires a GPU."
else:
    try:
        GPU = _probe_accelerator()
    except Exception as exc:
        _SKIP_REASON = (
            f"Accelerator probe failed ({type(exc).__name__}: {exc})."
        )

pytestmark = pytest.mark.skipif(
    _SKIP_REASON is not None, reason=_SKIP_REASON or ""
)


def _gpu() -> Accelerator:
    # Tests are skipped via pytestmark when GPU is None, so this only runs
    # when the probe succeeded. Narrows the Optional for the type checker.
    assert GPU is not None, "GPU should be set when tests run"
    return GPU


@pytest.fixture(scope="session", autouse=True)
def _require_warm_adoption() -> None:
    """Fail if the eager-GC warm cache didn't adopt.

    The forward-pass tests pass whether the warm adopts or silently cold-
    compiles, so this asserts adoption directly, as in test_interp_warm_cache
    and test_interpreter_ops_gpu.
    """
    if _SKIP_REASON is not None or not os.environ.get("XARCH_WARM_RLOCATION"):
        return
    unadopted = [
        f.name for f in GC_FAMILIES if not adopted_from_manifest(f.name)
    ]
    if unadopted:
        raise RuntimeError(
            f"eager-GC warm is wired but {unadopted} did not adopt from the"
            " manifest; a silent cold-compile fallback."
        )


# ----- Metal backend known-bug guard ---------------------------------------

_IS_METAL = GPU is not None and accelerator_api() == "metal"

# Marker for forward-pass tests that compare GPU output against CPU output.
# On Metal, the known seq_len > 3 codegen bug makes those comparisons fail
# with AssertionError; xfail catches that. strict=True ensures we get an
# XPASS-as-failure signal the day Metal stops producing wrong results, so
# we know to delete the marker.
xfail_on_metal_long_seq = pytest.mark.xfail(
    _IS_METAL,
    reason=(
        "Known Metal backend bug: incorrect codegen for seq_len > 3 "
        "produces value divergence vs CPU reference."
    ),
    raises=AssertionError,
    strict=True,
)


def _run_on_cpu_and_gpu(
    build: Callable[[], tuple[Module, Tensor]],  # type: ignore[type-arg]
) -> tuple[np.ndarray, np.ndarray]:
    """Build module + input on CPU, run; transfer module/input to GPU, run.

    Returns ``(cpu_out, gpu_out)`` as numpy arrays. The same materialized
    weight tensors are used on both devices (``Module.to(GPU)`` copies
    the realized weights), so the only intentional source of divergence
    is the backend kernel implementation.
    """
    with default_device(CPU()), default_dtype(DType.float32):
        module, x = build()
        cpu_out = np.from_dlpack(module(x).to(CPU()))

    gpu_module = module.to(_gpu())
    gpu_x = x.to(_gpu())
    with default_device(_gpu()), default_dtype(DType.float32):
        gpu_out = np.from_dlpack(gpu_module(gpu_x).to(CPU()))

    return cpu_out, gpu_out


# GPU vs CPU drift for single-module forward passes. Attention softmax on
# short inner axes (seq_len <= 32) can differ by ~3e-3 across backends.
_BLOCK_FORWARD_ATOL = 5e-3


class TestGPT2Config:
    """Test GPT2Config dataclass."""

    def test_default_values(self) -> None:
        config = GPT2Config()
        assert config.vocab_size == 50257
        assert config.n_positions == 1024
        assert config.n_embd == 768
        assert config.n_layer == 12
        assert config.n_head == 12
        assert config.n_inner is None
        assert config.layer_norm_epsilon == 1e-5

    def test_custom_values(self) -> None:
        config = GPT2Config()
        config.n_embd = 512
        config.n_layer = 6
        assert config.n_embd == 512
        assert config.n_layer == 6


class TestGPT2MLP:
    """Test GPT2MLP module on GPU."""

    def test_initialization(self) -> None:
        config = GPT2Config()
        mlp = GPT2MLP(intermediate_size=3072, config=config)
        assert mlp.c_fc is not None
        assert mlp.c_proj is not None

    @xfail_on_metal_long_seq
    def test_forward_pass_matches_cpu(self) -> None:
        config = GPT2Config()

        def build() -> tuple[GPT2MLP, Tensor]:
            mlp = GPT2MLP(intermediate_size=3072, config=config)
            x = Tensor.ones([2, 10, config.n_embd], dtype=DType.float32)
            return mlp, x

        cpu_out, gpu_out = _run_on_cpu_and_gpu(build)
        np.testing.assert_allclose(
            gpu_out, cpu_out, atol=_BLOCK_FORWARD_ATOL, rtol=1e-3
        )


class TestCausalMask:
    """Test causal_mask function on GPU.

    These checks operate on a constant-fill plus triangular indexing
    pattern, with no compute kernel that the Metal seq_len > 3 bug
    affects, so they run straight on Metal without xfail.
    """

    def test_causal_mask_shape(self) -> None:
        seq_len = 5
        mask = causal_mask(seq_len, 0, dtype=DType.float32, device=_gpu())
        assert [int(d) for d in mask.shape] == [seq_len, seq_len]

    def test_causal_mask_values(self) -> None:
        seq_len = 4
        mask = causal_mask(seq_len, 0, dtype=DType.float32, device=_gpu())

        mask_np = np.from_dlpack(mask.to(CPU()))

        for i in range(seq_len):
            for j in range(i + 1):
                assert mask_np[i, j] == 0.0
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                assert mask_np[i, j] == float("-inf")


class TestGPT2MultiHeadAttention:
    """Test GPT2MultiHeadAttention module on GPU."""

    def test_initialization(self) -> None:
        config = GPT2Config()
        attn = GPT2MultiHeadAttention(config)
        assert attn.embed_dim == config.n_embd
        assert attn.num_heads == config.n_head
        assert attn.head_dim == config.n_embd // config.n_head
        assert attn.c_attn is not None
        assert attn.c_proj is not None

    def test_split_heads_shape(self) -> None:
        # Pure reshape + transpose; no compute kernel that the Metal
        # seq_len > 3 bug would affect, so this runs on Metal too.
        config = GPT2Config()
        batch_size, seq_len = 2, 10
        with default_device(_gpu()), default_dtype(DType.float32):
            attn = GPT2MultiHeadAttention(config)
            tensor = Tensor.ones(
                [batch_size, seq_len, config.n_embd], dtype=DType.float32
            )
            split = attn._split_heads(tensor, config.n_head, attn.head_dim)
        assert [int(d) for d in split.shape] == [
            batch_size,
            config.n_head,
            seq_len,
            attn.head_dim,
        ]

    def test_merge_heads_shape(self) -> None:
        # Pure reshape + transpose; see test_split_heads_shape.
        config = GPT2Config()
        batch_size, seq_len = 2, 10
        with default_device(_gpu()), default_dtype(DType.float32):
            attn = GPT2MultiHeadAttention(config)
            tensor = Tensor.ones(
                [batch_size, config.n_head, seq_len, attn.head_dim],
                dtype=DType.float32,
            )
            merged = attn._merge_heads(tensor, config.n_head, attn.head_dim)
        assert [int(d) for d in merged.shape] == [
            batch_size,
            seq_len,
            config.n_embd,
        ]

    @xfail_on_metal_long_seq
    def test_forward_pass_matches_cpu(self) -> None:
        config = GPT2Config()

        def build() -> tuple[GPT2MultiHeadAttention, Tensor]:
            attn = GPT2MultiHeadAttention(config)
            x = Tensor.ones([2, 10, config.n_embd], dtype=DType.float32)
            return attn, x

        cpu_out, gpu_out = _run_on_cpu_and_gpu(build)
        np.testing.assert_allclose(
            gpu_out, cpu_out, atol=_BLOCK_FORWARD_ATOL, rtol=1e-3
        )


class TestLayerNorm:
    """Test LayerNorm module on GPU."""

    def test_initialization(self) -> None:
        dim = 768
        ln = LayerNorm(dim, eps=1e-5)
        assert ln.eps == 1e-5
        assert [int(d) for d in ln.weight.shape] == [dim]
        assert [int(d) for d in ln.bias.shape] == [dim]

    @xfail_on_metal_long_seq
    def test_forward_pass_matches_cpu(self) -> None:
        dim = 768

        def build() -> tuple[LayerNorm, Tensor]:
            ln = LayerNorm(dim)
            x = Tensor.ones([2, 10, dim], dtype=DType.float32)
            return ln, x

        cpu_out, gpu_out = _run_on_cpu_and_gpu(build)
        np.testing.assert_allclose(
            gpu_out, cpu_out, atol=_BLOCK_FORWARD_ATOL, rtol=1e-3
        )


class TestGPT2Block:
    """Test GPT2Block module on GPU."""

    def test_initialization(self) -> None:
        config = GPT2Config()
        block = GPT2Block(config)
        assert block.ln_1 is not None
        assert block.attn is not None
        assert block.ln_2 is not None
        assert block.mlp is not None

    def test_initialization_with_custom_inner_dim(self) -> None:
        config = GPT2Config()
        config.n_inner = 2048
        block = GPT2Block(config)
        assert block.mlp is not None

    @xfail_on_metal_long_seq
    def test_forward_pass_matches_cpu(self) -> None:
        config = GPT2Config()

        def build() -> tuple[GPT2Block, Tensor]:
            block = GPT2Block(config)
            x = Tensor.ones([2, 10, config.n_embd], dtype=DType.float32)
            return block, x

        cpu_out, gpu_out = _run_on_cpu_and_gpu(build)
        np.testing.assert_allclose(
            gpu_out, cpu_out, atol=_BLOCK_FORWARD_ATOL, rtol=1e-3
        )


class TestMaxGPT2Model:
    """Test MaxGPT2Model module on GPU."""

    def test_initialization(self) -> None:
        config = GPT2Config()
        model = MaxGPT2Model(config)
        assert model.wte is not None
        assert model.wpe is not None
        assert model.h is not None
        assert model.ln_f is not None

    @xfail_on_metal_long_seq
    def test_forward_pass_matches_cpu(self) -> None:
        config = GPT2Config()

        def build() -> tuple[MaxGPT2Model, Tensor]:
            model = MaxGPT2Model(config)
            input_ids = Tensor.zeros([2, 10], dtype=DType.int64)
            return model, input_ids

        cpu_out, gpu_out = _run_on_cpu_and_gpu(build)
        # Looser tolerance: fp32 noise accumulates over 12 transformer
        # blocks; near-zero output positions inflate relative error.
        # atol covers cross-backend drift (B200 ~0.02, MI355 ~0.12).
        np.testing.assert_allclose(gpu_out, cpu_out, atol=2e-1, rtol=1e-3)


class TestMaxGPT2LMHeadModel:
    """Test MaxGPT2LMHeadModel module on GPU."""

    def test_initialization(self) -> None:
        config = GPT2Config()
        model = MaxGPT2LMHeadModel(config)
        assert model.config == config
        assert model.transformer is not None
        assert model.lm_head is not None

    @xfail_on_metal_long_seq
    def test_forward_pass_matches_cpu(self) -> None:
        config = GPT2Config()

        def build() -> tuple[MaxGPT2LMHeadModel, Tensor]:
            model = MaxGPT2LMHeadModel(config)
            input_ids = Tensor.zeros([2, 10], dtype=DType.int64)
            return model, input_ids

        cpu_out, gpu_out = _run_on_cpu_and_gpu(build)
        # Same rationale as TestMaxGPT2Model: 12 blocks + lm_head
        # projection accumulates fp32 noise; near-zero outputs inflate
        # relative error. atol covers cross-backend drift.
        np.testing.assert_allclose(gpu_out, cpu_out, atol=2e-1, rtol=1e-3)


class TestModelDimensions:
    """Test that model dimensions are consistent throughout."""

    def test_head_dimensions(self) -> None:
        config = GPT2Config()
        assert config.n_embd % config.n_head == 0
        head_dim = config.n_embd // config.n_head
        assert head_dim == 64

    def test_mlp_inner_dimension(self) -> None:
        config = GPT2Config()
        expected_inner = 4 * config.n_embd
        assert expected_inner == 3072

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
"""Checks the minillm package on a tiny random model, without downloads."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pytest
from max import functional as F
from max.dtype import DType
from max.tensor import Tensor, default_device, default_dtype
from max.tensor.tensor import TensorType
from minillm.attention import Attention, causal_mask, naive_attention
from minillm.config import LlamaConfig
from minillm.device import pick_device
from minillm.engine import Engine
from minillm.kv_cache import CachedLlamaForCausalLM, KVCacheDecoder
from minillm.model import LlamaForCausalLM
from minillm.paged import BlockPool, PagedEngine
from minillm.prefix import PrefixCachingEngine, PrefixCachingPool
from minillm.sampling import generate, sample
from minillm.step import (
    ContiguousCache,
    Request,
    StepModel,
    compile_step,
    run_step,
)
from minillm.weights import read_safetensors

DEVICE = pick_device()
# Compiling a model takes minutes under bazel's debug build of MAX, so bazel
# sets MINILLM_SKIP_COMPILES and runs only the other checks.
compiles = pytest.mark.skipif(
    os.environ.get("MINILLM_SKIP_COMPILES") == "1", reason="compiles a model"
)
TINY = LlamaConfig(
    vocab_size=64,
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    rope_theta=10000.0,
    max_position_embeddings=128,
)
PROMPT = [1, 5, 9, 3, 7]


def tokens(ids: list[int]) -> Tensor:
    return Tensor(np.array([ids], dtype=np.int64), device=DEVICE)


@pytest.fixture(scope="module")
def model() -> LlamaForCausalLM:
    with default_device(DEVICE), default_dtype(DType.float32):
        return LlamaForCausalLM(TINY)


@pytest.fixture(scope="module")
def compiled(model: LlamaForCausalLM):  # noqa: ANN201
    return model.compile(
        TensorType(DType.int64, ("batch", "seq_len"), device=DEVICE)
    )


@pytest.fixture(scope="module")
def step(model: LlamaForCausalLM):  # noqa: ANN201
    with F.lazy(), default_device(DEVICE), default_dtype(DType.float32):
        step_model = StepModel(TINY)
    step_model.load_state_dict({n: p.to_numpy() for n, p in model.parameters})
    return compile_step(step_model, TINY, DEVICE)


@compiles
def test_compiled_matches_eager(model: LlamaForCausalLM, compiled) -> None:  # noqa: ANN001
    eager = model(tokens(PROMPT)).to_numpy()
    assert eager.shape == (1, len(PROMPT), TINY.vocab_size)
    np.testing.assert_allclose(
        compiled(tokens(PROMPT)).to_numpy(), eager, rtol=1e-5, atol=1e-4
    )


@compiles
def test_flash_attention_matches_naive() -> None:
    with default_device(DEVICE), default_dtype(DType.float32):
        attn = Attention(TINY)
        x = Tensor(
            np.random.default_rng(0)
            .standard_normal((1, 7, TINY.hidden_size))
            .astype(np.float32),
            device=DEVICE,
        )
    q, k, v = attn.project(x, 0)
    naive = naive_attention(q, k, v, causal_mask(7, 7, DEVICE))
    expected = attn.o_proj(naive.reshape([1, 7, TINY.hidden_size]))
    # Random weights give outputs near 100, so compare relative error.
    np.testing.assert_allclose(
        attn(x).to_numpy(), expected.to_numpy(), rtol=1e-5, atol=1e-4
    )


def test_causal_mask_blocks_the_future() -> None:
    mask = causal_mask(3, 5, DEVICE).to_numpy()
    expected = np.array(
        [[0, 0, 0, -np.inf, -np.inf], [0, 0, 0, 0, -np.inf], [0, 0, 0, 0, 0]]
    )
    np.testing.assert_array_equal(mask, expected)


@compiles
def test_kv_cache_decode_matches_recompute(
    model: LlamaForCausalLM,
    compiled,  # noqa: ANN001
) -> None:
    with F.lazy(), default_device(DEVICE), default_dtype(DType.float32):
        cached_model = CachedLlamaForCausalLM(TINY)
    cached_model.load_state_dict(
        {name: p.to_numpy() for name, p in model.parameters}
    )
    decoder = KVCacheDecoder(cached_model, TINY, DEVICE)

    def recompute(ids: list[int]) -> Tensor:
        return compiled(tokens(ids))[0, -1]

    assert generate(decoder, PROMPT, 8, temperature=0) == generate(
        recompute, PROMPT, 8, temperature=0
    )
    np.testing.assert_allclose(
        decoder(PROMPT).to_numpy(), recompute(PROMPT).to_numpy(), atol=1e-4
    )


@compiles
def test_sample_filters_reduce_to_greedy() -> None:
    logits = Tensor(np.array([0.1, 3.0, 2.9, -1.0], np.float32), device=DEVICE)
    assert sample(logits, temperature=0) == 1
    assert sample(logits, top_k=1) == 1
    assert sample(logits, top_p=1e-6) == 1


def test_read_safetensors_widens_bfloat16(tmp_path: Path) -> None:
    values = np.array([1.5, -2.25, 3.0e-3], dtype=np.float32)
    bf16 = (values.view(np.uint32) >> 16).astype(np.uint16)  # exact here
    ints = np.arange(4, dtype=np.int64)
    header = {
        "w": {"dtype": "BF16", "shape": [3], "data_offsets": [0, 6]},
        "n": {"dtype": "I64", "shape": [2, 2], "data_offsets": [6, 38]},
    }
    raw = json.dumps(header).encode()
    path = tmp_path / "model.safetensors"
    path.write_bytes(
        len(raw).to_bytes(8, "little") + raw + bf16.tobytes() + ints.tobytes()
    )
    loaded = read_safetensors(str(path))
    expected = (bf16.astype(np.uint32) << 16).view(np.float32)
    np.testing.assert_array_equal(loaded["w"], expected)
    assert loaded["w"].dtype == np.float32
    np.testing.assert_array_equal(loaded["n"], ints.reshape(2, 2))
    assert math.isclose(float(loaded["w"][0]), 1.5)


@compiles
def test_padded_step_with_past_matches_single_sequences(step) -> None:  # noqa: ANN001
    # A shorter suffix is left-padded; with a non-empty past, a padding row
    # whose first key block is fully masked must not produce NaN.
    prefixes = [[3] * 300, [4] * 300]
    suffixes = [[5, 6, 7], [8, 9]]

    def last_logits(
        batch_suffixes: list[list[int]], rows: list[int]
    ) -> np.ndarray:
        caches = [ContiguousCache(TINY) for _ in rows]
        run_step(step, caches, [prefixes[i] for i in rows], DEVICE)
        return run_step(step, caches, batch_suffixes, DEVICE)

    batched = last_logits(suffixes, [0, 1])
    assert np.isfinite(batched).all()
    for i in range(2):
        alone = last_logits([suffixes[i]], [i])[0]
        np.testing.assert_allclose(batched[i], alone, rtol=1e-4, atol=1e-4)


@compiles
@pytest.mark.parametrize(
    ("engine_cls", "pool_cls"),
    [(PagedEngine, BlockPool), (PrefixCachingEngine, PrefixCachingPool)],
)
def test_paged_engine_preempts_and_recomputes(
    step,  # noqa: ANN001
    engine_cls: type[PagedEngine],
    pool_cls: type[BlockPool],
) -> None:
    def workload() -> list[Request]:
        return [Request([1, 2, 3] * k, 12) for k in (4, 8, 1, 2)]

    expected = workload()
    engine = Engine(step, TINY, DEVICE, max_batch=4)
    for request in expected:
        engine.add(request)
    engine.run()

    # The first two prompts fill the pool, so decoding must preempt one.
    pool = pool_cls(TINY, num_blocks=9, block_size=4)
    paged = engine_cls(step, TINY, DEVICE, max_batch=4, pool=pool)
    requests = workload()
    for request in requests:
        paged.add(request)
    paged.run()

    assert paged.preemptions > 0
    assert [r.output for r in requests] == [r.output for r in expected]
    starts = [s for r in requests if (s := r.first_token) is not None]
    assert len(starts) == len(requests)
    assert starts == sorted(starts)
    assert pool.available() == pool.num_blocks

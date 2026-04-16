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
"""GPT2PipelineModel — wraps the GPT-2 experimental model for max serve.

This model is intentionally naive: it does NOT use the KV cache (the full
token sequence is passed on every decode step). This makes the implementation
clear and teachable at the cost of efficiency. A production implementation
would use paged attention with incremental KV updates.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import max.experimental.functional as F
import numpy as np
from max.driver import CPU, Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession
from max.experimental.tensor import (
    Tensor,
    TensorType,
    default_device,
    default_dtype,
)
from max.graph import DeviceRef
from max.graph.weights import Weights, WeightsAdapter
from max.nn.kv_cache import KVCacheInputs, KVCacheParams
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.core import TextContext
from max.pipelines.lib import (
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
    PipelineModelWithKVCache,
    upper_bounded_default,
)
from max.pipelines.lib.utils import parse_state_dict_from_weights

from .gpt2 import MaxGPT2LMHeadModel


@dataclass
class GPT2Inputs(ModelInputs):
    """Model inputs for GPT-2."""

    tokens: Buffer
    input_row_offsets: Buffer

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        return (self.tokens,)


class GPT2PipelineModel(PipelineModelWithKVCache[TextContext]):
    """GPT-2 pipeline model for max serve.

    Uses max.experimental.nn — the same API as the tutorial's gpt2.py.
    Passes the full token sequence on every step (no incremental KV cache).
    This keeps the implementation simple and readable for tutorial purposes.
    """

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        devices: list[Device],
        kv_cache_config: KVCacheConfig,
        weights: Weights,
        adapter: WeightsAdapter | None,
        return_logits: ReturnLogits,
        return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE,
    ) -> None:
        super().__init__(
            pipeline_config=pipeline_config,
            session=session,
            devices=devices,
            kv_cache_config=kv_cache_config,
            weights=weights,
            adapter=adapter,
            return_logits=return_logits,
            return_hidden_states=return_hidden_states,
        )
        self.model = self._load_model(weights, adapter)

    @classmethod
    def get_kv_params(
        cls,
        huggingface_config: Any,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        return kv_cache_config.to_params(
            dtype=cache_dtype,
            n_kv_heads=huggingface_config.n_head,
            head_dim=huggingface_config.n_embd // huggingface_config.n_head,
            num_layers=huggingface_config.n_layer,
            devices=devices,
            data_parallel_degree=pipeline_config.model.data_parallel_degree,
        )

    @classmethod
    def calculate_max_seq_len(
        cls,
        pipeline_config: PipelineConfig,
        huggingface_config: Any,
    ) -> int:
        return upper_bounded_default(
            upper_bound=huggingface_config.n_positions,
            default=pipeline_config.model.max_length,
        )

    # ANCHOR: load_model
    def _load_model(
        self,
        weights: Weights,
        adapter: WeightsAdapter | None,
    ) -> Any:
        hf_config = self.huggingface_config
        device = self.devices[0]

        state_dict = parse_state_dict_from_weights(
            self.pipeline_config, weights, adapter
        )

        with F.lazy(), default_device(device), default_dtype(self.dtype):
            gpt2_module = MaxGPT2LMHeadModel(hf_config)
            gpt2_module.to(device)

        token_type = TensorType(
            DType.int64, ("batch", "seq_len"), device=device
        )
        return gpt2_module.compile(token_type, weights=state_dict)

    # ANCHOR_END: load_model

    # ANCHOR: execute_method
    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        assert isinstance(model_inputs, GPT2Inputs)

        input_tensor = Tensor.from_dlpack(model_inputs.tokens).to(
            self.devices[0]
        )

        all_logits: Tensor = self.model(input_tensor)

        last_logits_np: np.ndarray = np.from_dlpack(all_logits.to(CPU()))
        last_logits_np = np.ascontiguousarray(last_logits_np[0, -1:, :])

        last_buf = Buffer.from_numpy(last_logits_np).to(self.devices[0])
        return ModelOutputs(logits=last_buf, next_token_logits=last_buf)

    # ANCHOR_END: execute_method

    # ANCHOR: token_inputs
    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputs[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> GPT2Inputs:
        _ = return_n_logits  # PipelineModel API; last-token logits only in `execute`.
        ctx = replica_batches[0][0]
        token_ids = _tokens_from_context(ctx)
        inputs = _make_gpt2_inputs(token_ids, self.devices[0])
        inputs.kv_cache_inputs = kv_cache_inputs
        return inputs

    def prepare_next_token_inputs(
        self,
        next_tokens: Buffer,
        prev_model_inputs: ModelInputs,
    ) -> GPT2Inputs:
        assert isinstance(prev_model_inputs, GPT2Inputs)
        prev_np: np.ndarray = np.from_dlpack(prev_model_inputs.tokens.to(CPU()))
        new_token_np: np.ndarray = np.from_dlpack(next_tokens.to(CPU()))
        new_token = int(new_token_np.ravel()[0])
        extended = np.concatenate([prev_np.ravel(), [new_token]])[np.newaxis, :]
        inputs = _make_gpt2_inputs(extended.ravel().tolist(), self.devices[0])
        inputs.kv_cache_inputs = prev_model_inputs.kv_cache_inputs
        return inputs

    # ANCHOR_END: token_inputs


def _tokens_from_context(ctx: TextContext) -> list[int]:
    """Return every token ID (prompt + generated) for full-sequence replay.

    ``TextContext.tokens`` is a ``TokenBuffer``; ``all`` is the full history,
    which this no-incremental-KV model must rerun on each decode step.
    """
    return ctx.tokens.all.tolist()


def _make_gpt2_inputs(token_ids: list[int], device: Device) -> GPT2Inputs:
    tokens_np = np.array([token_ids], dtype=np.int64)
    tokens_buf = Buffer.from_numpy(tokens_np).to(device)
    # Ragged row offsets: [0, seq_len] for a single sequence.
    row_offsets_np = np.array([0, len(token_ids)], dtype=np.uint32)
    row_offsets_buf = Buffer.from_numpy(row_offsets_np)
    return GPT2Inputs(tokens=tokens_buf, input_row_offsets=row_offsets_buf)

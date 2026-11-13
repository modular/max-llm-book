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
"""Generated from 07_01_serve_your_model.ipynb by tools/nb2book.py.

Edit the notebook, then run ``pixi run export``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from max.driver import Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef
from max.graph.weights import Weights, WeightsAdapter, WeightsFormat
from max.nn.kv_cache import KVCacheInputs, KVCacheParams
from max.nn.outputs import ReturnHiddenStates, ReturnLogits
from max.pipelines.context import TextContext
from max.pipelines.lib import (
    KVCacheConfig,
    MAXModelConfig,
    MemoryPlan,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
    PipelineModelWithKVCache,
    SupportedArchitecture,
    SupportedEncoding,
    TextTokenizer,
)
from max.pipelines.lib.interfaces.arch_config import (
    ArchConfigWithAttentionKVCache,
)
from max.pipelines.lib.utils import (
    parse_state_dict_from_weights,
    upper_bounded_default,
)
from max.pipelines.modeling.types import PipelineTask
from max.tensor import Tensor
from max.tensor.tensor import TensorType
from transformers import AutoConfig

from minillm.config import LlamaConfig
from minillm.model import load_model


@dataclass
class MinillmArchConfig(ArchConfigWithAttentionKVCache):
    DEFAULT_ENCODING: ClassVar[SupportedEncoding] = "float32"
    SUPPORTED_ENCODINGS: ClassVar[set[SupportedEncoding]] = {"float32"}

    @property
    def num_key_value_heads(self) -> int:
        return self.huggingface_config.num_key_value_heads

    @property
    def head_dim(self) -> int:
        hf = self.huggingface_config
        return (
            getattr(hf, "head_dim", None)
            or hf.hidden_size // hf.num_attention_heads
        )

    @property
    def num_layers(self) -> int:
        return self.huggingface_config.num_hidden_layers

    @classmethod
    def calculate_max_seq_len(
        cls, huggingface_config: AutoConfig, model_config: MAXModelConfig
    ) -> int:
        return upper_bounded_default(
            upper_bound=huggingface_config.max_position_embeddings,
            default=model_config.max_length,
        )


@dataclass
class TokenInputs(ModelInputs):
    tokens: Buffer

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        return (self.tokens,)


class MinillmPipelineModel(PipelineModelWithKVCache[TextContext]):
    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        devices: list[Device],
        kv_cache_config: KVCacheConfig,
        weights: Weights,
        adapter: WeightsAdapter | None,
        return_logits: ReturnLogits,
        *,
        memory_plan: MemoryPlan,
        return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE,
        max_batch_size: int = 1,
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
            max_batch_size=max_batch_size,
            memory_plan=memory_plan,
        )
        hf = self.huggingface_config
        config = LlamaConfig(
            **{
                k: getattr(hf, k)
                for k in LlamaConfig.__dataclass_fields__
                if hasattr(hf, k)
            }
        )
        state = parse_state_dict_from_weights(
            self.pipeline_config, weights, adapter
        )
        state = {name: w.astype(DType.float32) for name, w in state.items()}
        model = load_model(config, state, self.devices[0])
        self.model = model.compile(
            TensorType(
                DType.int64, ("batch", "seq_len"), device=self.devices[0]
            )
        )

    @classmethod
    def get_kv_params(
        cls,
        huggingface_config: Any,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        hf = huggingface_config
        return kv_cache_config.to_params(
            dtype=cache_dtype,
            n_kv_heads=hf.num_key_value_heads,
            head_dim=getattr(hf, "head_dim", None)
            or hf.hidden_size // hf.num_attention_heads,
            num_layers=hf.num_hidden_layers,
            devices=devices,
            data_parallel_degree=pipeline_config.model.data_parallel_degree,
        )

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        assert isinstance(model_inputs, TokenInputs)
        logits = self.model(Tensor.from_dlpack(model_inputs.tokens))
        last = Buffer.from_numpy(
            np.ascontiguousarray(logits.to_numpy()[0, -1:, :])
        ).to(self.devices[0])
        return ModelOutputs(logits=last, next_token_logits=last)

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputs[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> TokenInputs:
        ids = replica_batches[0][0].tokens.all
        inputs = TokenInputs(
            tokens=Buffer.from_numpy(np.array([ids], dtype=np.int64)).to(
                self.devices[0]
            )
        )
        inputs.kv_cache_inputs = kv_cache_inputs
        return inputs


def _identity_adapter(state_dict: dict[str, Weights], **_: object) -> dict:
    return {name: w.data() for name, w in state_dict.items()}


minillm_arch = SupportedArchitecture(
    name="LlamaForCausalLM",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=["HuggingFaceTB/SmolLM2-135M-Instruct"],
    default_weights_format=WeightsFormat.safetensors,
    default_encoding=MinillmArchConfig.DEFAULT_ENCODING,
    supported_encodings=MinillmArchConfig.SUPPORTED_ENCODINGS,
    pipeline_model=MinillmPipelineModel,
    tokenizer=TextTokenizer,
    context_type=TextContext,
    multi_gpu_supported=False,
    weight_adapters={WeightsFormat.safetensors: _identity_adapter},
    config=MinillmArchConfig,
    required_arguments={"enable_prefix_caching": False, "max_batch_size": 1},
    supports_overlap_scheduler=False,
    supports_device_graph_capture=False,
)
ARCHITECTURES = [minillm_arch]

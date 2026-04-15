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
# gpt2_arch/arch.py
"""SupportedArchitecture registration for GPT-2.

Pass `--custom-architectures gpt2_arch` to `max serve`:
    max serve --custom-architectures gpt2_arch --model gpt2
"""

# ANCHOR: book
from __future__ import annotations

from max.graph.weights import WeightsFormat
from max.interfaces import PipelineTask
from max.pipelines.core import TextContext
from max.pipelines.lib import SupportedArchitecture, TextTokenizer

from . import weight_adapters
from .model import GPT2PipelineModel
from .model_config import GPT2ArchConfig

gpt2_arch = SupportedArchitecture(
    # Must match the HuggingFace config "architectures" field
    name="GPT2LMHeadModel",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=["gpt2", "openai-community/gpt2"],
    default_weights_format=WeightsFormat.safetensors,
    default_encoding="float32",
    supported_encodings={"float32"},
    pipeline_model=GPT2PipelineModel,
    tokenizer=TextTokenizer,
    context_type=TextContext,
    multi_gpu_supported=False,
    rope_type="none",
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict,
    },
    config=GPT2ArchConfig,
    required_arguments={"enable_prefix_caching": False},
)
# ANCHOR_END: book

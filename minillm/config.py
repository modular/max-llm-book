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
"""Generated from 01_03_tokens_and_weights.ipynb by tools/nb2book.py.

Edit the notebook, then run ``pixi run export``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path

from huggingface_hub import hf_hub_download

MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"


@dataclass(frozen=True)
class LlamaConfig:
    """The sizes of a Llama-family model, as in its ``config.json``."""

    vocab_size: int = 49152
    hidden_size: int = 576
    intermediate_size: int = 1536
    num_hidden_layers: int = 30
    num_attention_heads: int = 9
    num_key_value_heads: int = 3
    rope_theta: float = 100000.0
    rms_norm_eps: float = 1e-5
    max_position_embeddings: int = 8192
    tie_word_embeddings: bool = True
    eos_token_id: int = 2
    head_dim: int = 0  # 0 means hidden_size // num_attention_heads

    def __post_init__(self) -> None:
        if (
            not self.head_dim
        ):  # newer configs, such as Qwen3's, give it explicitly
            object.__setattr__(
                self, "head_dim", self.hidden_size // self.num_attention_heads
            )

    @classmethod
    def from_pretrained(cls, model_id: str = MODEL_ID) -> LlamaConfig:
        """Reads the config of a Hugging Face model."""
        raw = json.loads(
            Path(hf_hub_download(model_id, "config.json")).read_text()
        )
        return cls(
            **{f.name: raw[f.name] for f in fields(cls) if f.name in raw}
        )

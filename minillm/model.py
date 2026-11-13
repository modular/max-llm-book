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
"""Generated from 01_06_full_model.ipynb by tools/nb2book.py.

Edit the notebook, then run ``pixi run export``.
"""

from __future__ import annotations

from max import functional as F
from max.driver import Device
from max.dtype import DType
from max.nn import Embedding, Linear, Module, ModuleList, RMSNorm
from max.tensor import Tensor, default_device, default_dtype
from numpy.typing import NDArray

from minillm.attention import Attention
from minillm.config import LlamaConfig
from minillm.layers import MLP


class DecoderLayer(Module):
    """One transformer layer: attention and an MLP, each with a residual."""

    def __init__(self, config: LlamaConfig) -> None:
        eps = config.rms_norm_eps
        self.input_layernorm = RMSNorm(config.hidden_size, eps=eps)
        self.self_attn = Attention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=eps)
        self.mlp = MLP(config)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.self_attn(self.input_layernorm(x))
        return x + self.mlp(self.post_attention_layernorm(x))


class LlamaModel(Module):
    """The embedding, the decoder layers, and the final norm."""

    def __init__(self, config: LlamaConfig) -> None:
        self.embed_tokens = Embedding(config.vocab_size, dim=config.hidden_size)
        self.layers = ModuleList(
            [DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: Tensor) -> Tensor:
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class LlamaForCausalLM(Module):
    """A Llama model with its language-model head: token IDs in, logits out."""

    def __init__(self, config: LlamaConfig) -> None:
        self.model = LlamaModel(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids: Tensor) -> Tensor:
        return self.lm_head(self.model(input_ids))


def load_model(
    config: LlamaConfig,
    weights: dict[str, NDArray],
    device: Device,
    model_class: type[Module] = LlamaForCausalLM,
) -> Module:
    """Builds ``model_class`` on ``device`` and loads ``weights`` into it."""
    with F.lazy(), default_device(device), default_dtype(DType.float32):
        model = model_class(config)
    if config.tie_word_embeddings:
        weights = {
            **weights,
            "lm_head.weight": weights["model.embed_tokens.weight"],
        }
    model.load_state_dict(weights)
    return model

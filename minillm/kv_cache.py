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
"""Generated from 01_08_kv_cache.ipynb by tools/nb2book.py.

Edit the notebook, then run ``pixi run export``.
"""

from __future__ import annotations

import math

import numpy as np
from max import functional as F
from max.driver import Device
from max.dtype import DType
from max.graph import TensorType
from max.nn import ModuleList
from max.nn.kernels import flash_attention
from max.tensor import Tensor

from minillm.attention import Attention
from minillm.config import LlamaConfig
from minillm.model import DecoderLayer, LlamaForCausalLM, LlamaModel

KVPair = tuple[Tensor, Tensor]


class CachedAttention(Attention):
    """Attention that reads and extends a key/value cache."""

    def forward(
        self, x: Tensor, start: Tensor, past: KVPair | None = None
    ) -> tuple[Tensor, KVPair]:
        q, k, v = self.project(x, start)
        if past is None:  # prefill: causal attention over the prompt
            return self.attend(q, k, v), (k, v)
        k = F.concat([past[0], k], axis=1)
        v = F.concat([past[1], v], axis=1)
        # The one new token may attend to every cached position: no mask.
        out = flash_attention(q, k, v, 1 / math.sqrt(self.head_dim))
        out = out.rebind_mapping(q.mapping)
        batch, seq = q.shape[0], q.shape[1]
        out = out.reshape([batch, seq, self.num_heads * self.head_dim])
        return self.o_proj(out), (k, v)


class CachedDecoderLayer(DecoderLayer):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__(config)
        self.self_attn = CachedAttention(config)

    def forward(
        self, x: Tensor, start: Tensor, past: KVPair | None = None
    ) -> tuple[Tensor, KVPair]:
        attn, kv = self.self_attn(self.input_layernorm(x), start, past)
        x = x + attn
        return x + self.mlp(self.post_attention_layernorm(x)), kv


class CachedLlamaModel(LlamaModel):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__(config)
        self.layers = ModuleList(
            [
                CachedDecoderLayer(config)
                for _ in range(config.num_hidden_layers)
            ]
        )

    def forward(
        self,
        input_ids: Tensor,
        start: Tensor,
        cache: list[KVPair] | None = None,
    ) -> tuple[Tensor, list[KVPair]]:
        x = self.embed_tokens(input_ids)
        new_cache = []
        for i, layer in enumerate(self.layers):
            x, kv = layer(x, start, None if cache is None else cache[i])
            new_cache.append(kv)
        return self.norm(x), new_cache


class CachedLlamaForCausalLM(LlamaForCausalLM):
    """Returns the last position's logits and the updated cache."""

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__(config)
        self.model = CachedLlamaModel(config)

    def forward(
        self,
        input_ids: Tensor,
        start: Tensor,
        cache: list[KVPair] | None = None,
    ) -> tuple[Tensor, list[KVPair]]:
        hidden, new_cache = self.model(input_ids, start, cache)
        return self.lm_head(hidden[:, -1]), new_cache


class KVCacheDecoder:
    """Returns next-token logits like a full forward pass, using a cache."""

    def __init__(
        self, model: CachedLlamaForCausalLM, config: LlamaConfig, device: Device
    ) -> None:
        self.device = device
        tokens = TensorType(DType.int64, ("batch", "seq_len"), device=device)
        one_token = TensorType(DType.int64, ("batch", 1), device=device)
        start = TensorType(DType.float32, [1], device=device)
        kv = TensorType(
            DType.float32,
            ("batch", "cache_len", config.num_key_value_heads, config.head_dim),
            device=device,
        )
        self.prefill = model.compile(tokens, start)
        self.decode = model.compile(
            one_token, start, [(kv, kv)] * config.num_hidden_layers
        )
        self.cache: list[KVPair] | None = None
        self.length = 0

    def __call__(self, ids: list[int]) -> Tensor:
        if self.cache is not None and len(ids) == self.length + 1:
            logits, self.cache = self.decode(
                self._tokens(ids[-1:]), self._position(self.length), self.cache
            )
        else:
            logits, self.cache = self.prefill(
                self._tokens(ids), self._position(0)
            )
        self.length = len(ids)
        return logits[0]

    def _tokens(self, ids: list[int]) -> Tensor:
        return Tensor(np.array([ids], dtype=np.int64), device=self.device)

    def _position(self, position: int) -> Tensor:
        return Tensor(
            np.array([position], dtype=np.float32), device=self.device
        )

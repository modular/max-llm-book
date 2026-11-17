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
"""Generated from 05_01_request_lifecycle.ipynb by tools/nb2book.py.

Edit the notebook, then run ``pixi run export``.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from max import functional as F
from max.dtype import DType
from max.nn import ModuleList
from max.nn.kernels.attention import flash_attention
from max.tensor import Tensor

from minillm.attention import Attention
from minillm.config import LlamaConfig
from minillm.layers import rotate_half
from minillm.model import DecoderLayer, LlamaForCausalLM

KVPair = tuple[Tensor, Tensor]
Step = Callable[..., tuple[Tensor, list[KVPair]]]
# Masked scores use a large finite value instead of -inf: attention kernels
# process keys in blocks, and a row whose first block is fully -inf computes
# exp(-inf - -inf) = NaN, which then spreads through the values to other rows.
MASKED = -1e30


def rope_at(
    x: Tensor, positions: Tensor, head_dim: int, theta: float
) -> Tensor:
    """Rotates ``x``, ``[batch, q, heads, head_dim]``, for per-token
    ``positions``, ``[batch, q]``."""
    inv_freq = Tensor(
        [theta ** (-i / head_dim) for i in range(0, head_dim, 2)],
        dtype=DType.float32,
        device=x.mesh,
    )
    batch, q = positions.shape
    angles = positions.reshape([batch, q, 1]) * inv_freq
    angles = F.concat([angles, angles], axis=-1).reshape(
        [batch, q, 1, head_dim]
    )
    return x * F.cos(angles) + rotate_half(x) * F.sin(angles)


class StepAttention(Attention):
    """Attends from new tokens to the past keys and values plus their own."""

    def forward(
        self, x: Tensor, positions: Tensor, mask: Tensor, past: KVPair
    ) -> tuple[Tensor, KVPair]:
        batch, q, _ = x.shape
        query = self.q_proj(x).reshape(
            [batch, q, self.num_heads, self.head_dim]
        )
        key = self.k_proj(x).reshape(
            [batch, q, self.num_kv_heads, self.head_dim]
        )
        value = self.v_proj(x).reshape(
            [batch, q, self.num_kv_heads, self.head_dim]
        )
        query = rope_at(query, positions, self.head_dim, self.rope_theta)
        key = rope_at(key, positions, self.head_dim, self.rope_theta)
        all_keys = F.concat([past[0], key], axis=1)
        all_values = F.concat([past[1], value], axis=1)
        # The mask has one column per past and new position.
        mask = F.rebind(mask, [batch, q, all_keys.shape[1]])
        out = flash_attention(
            query, all_keys, all_values, 1 / math.sqrt(self.head_dim), mask
        ).rebind_mapping(query.mapping)
        out = out.reshape([batch, q, self.num_heads * self.head_dim])
        return self.o_proj(out), (key, value)


class StepLayer(DecoderLayer):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__(config)
        self.self_attn = StepAttention(config)

    def forward(
        self, x: Tensor, positions: Tensor, mask: Tensor, past: KVPair
    ) -> tuple[Tensor, KVPair]:
        attn, kv = self.self_attn(
            self.input_layernorm(x), positions, mask, past
        )
        x = x + attn
        return x + self.mlp(self.post_attention_layernorm(x)), kv


class StepModel(LlamaForCausalLM):
    """Advances a batch of sequences: returns each one's next-token logits
    and the new tokens' keys and values for every layer."""

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__(config)
        self.model.layers = ModuleList(
            [StepLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        tokens: Tensor,
        positions: Tensor,
        mask: Tensor,
        past: list[KVPair],
    ) -> tuple[Tensor, list[KVPair]]:
        x = self.model.embed_tokens(tokens)
        new_kv = []
        for layer, layer_past in zip(self.model.layers, past, strict=True):
            x, kv = layer(x, positions, mask, layer_past)
            new_kv.append(kv)
        return self.lm_head(self.model.norm(x)[:, -1]), new_kv


import numpy as np
from max.driver import Device
from max.tensor.tensor import TensorType


def compile_step(model: StepModel, config: LlamaConfig, device: Device) -> Step:
    """Compiles ``model`` for any batch size, number of new tokens, and past
    length."""
    kv = TensorType(
        DType.float32,
        ("batch", "past", config.num_key_value_heads, config.head_dim),
        device=device,
    )
    return model.compile(
        TensorType(DType.int64, ("batch", "q"), device=device),
        TensorType(DType.float32, ("batch", "q"), device=device),
        TensorType(DType.float32, ("batch", "q", "kv"), device=device),
        [(kv, kv)] * config.num_hidden_layers,
    )


class ContiguousCache:
    """One sequence's keys and values: a growing array per layer."""

    def __init__(self, config: LlamaConfig) -> None:
        shape = (0, config.num_key_value_heads, config.head_dim)
        self.layers = [
            [np.zeros(shape, np.float32), np.zeros(shape, np.float32)]
            for _ in range(config.num_hidden_layers)
        ]

    @property
    def length(self) -> int:
        return len(self.layers[0][0])

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    def read(self, layer: int) -> tuple[np.ndarray, np.ndarray]:
        return self.layers[layer][0], self.layers[layer][1]

    def append(self, layer: int, keys: np.ndarray, values: np.ndarray) -> None:
        entry = self.layers[layer]
        entry[0] = np.concatenate([entry[0], keys])
        entry[1] = np.concatenate([entry[1], values])


def run_step(
    step: Step, caches: list, new_tokens: list[list[int]], device: Device
) -> np.ndarray:
    """Advances each sequence by its ``new_tokens``, appends their keys and
    values to its cache, and returns next-token logits, ``[batch, vocab]``."""
    batch, q = len(caches), max(len(t) for t in new_tokens)
    past = max(c.length for c in caches)
    tokens = np.zeros((batch, q), np.int64)
    positions = np.zeros((batch, q), np.float32)
    mask = np.full((batch, q, past + q), MASKED, np.float32)
    for i, (cache, new) in enumerate(zip(caches, new_tokens, strict=True)):
        pad, n = q - len(new), cache.length
        tokens[i, pad:] = new
        positions[i, pad:] = np.arange(n, n + len(new))
        mask[i, pad:, :n] = 0.0  # every new token sees the sequence's past
        mask[i, pad:, past + pad :] = np.triu(
            np.full((len(new), len(new)), MASKED), 1
        )
        mask[i, np.arange(pad), past + np.arange(pad)] = (
            0.0  # padding sees itself
        )
    past_kv = []
    for layer in range(caches[0].num_layers):
        keys = np.zeros(
            (batch, past, *caches[0].read(layer)[0].shape[1:]), np.float32
        )
        values = np.zeros_like(keys)
        for i, cache in enumerate(caches):
            k, v = cache.read(layer)
            keys[i, : len(k)], values[i, : len(v)] = k, v
        past_kv.append(
            (Tensor(keys, device=device), Tensor(values, device=device))
        )
    logits, new_kv = step(
        Tensor(tokens, device=device),
        Tensor(positions, device=device),
        Tensor(mask, device=device),
        past_kv,
    )
    for layer, (k, v) in enumerate(new_kv):
        k, v = k.to_numpy(), v.to_numpy()
        for i, (cache, new) in enumerate(zip(caches, new_tokens, strict=True)):
            pad = q - len(new)
            cache.append(layer, k[i, pad:], v[i, pad:])
    return logits.to_numpy()


from dataclasses import dataclass, field


@dataclass
class Request:
    prompt: list[int]
    max_new_tokens: int
    output: list[int] = field(default_factory=list)
    arrived: float = 0.0
    first_token: float | None = None
    finished: float | None = None

    @property
    def done(self) -> bool:
        return len(self.output) >= self.max_new_tokens


import time
from collections.abc import Mapping

Tokenizer = Callable[[list[str]], Mapping[str, list[list[int]]]]

WORKLOAD = [
    ("Tell me about Paris.", 32),
    ("Write a haiku about the sea.", 24),
    ("What is 2 + 2?", 8),
    ("List three colors.", 16),
    ("Explain gravity in one sentence.", 32),
    ("Name a fruit.", 8),
    ("Describe a cat.", 24),
    ("Say hello in French.", 16),
]


def make_workload(tokenizer: Tokenizer) -> list[Request]:
    """Returns the chapter's eight requests, all arriving now."""
    now = time.perf_counter()
    return [
        Request(tokenizer([text])["input_ids"][0], n, arrived=now)
        for text, n in WORKLOAD
    ]

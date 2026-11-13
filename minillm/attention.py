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
"""Generated from 01_05_attention.ipynb by tools/nb2book.py.

Edit the notebook, then run ``pixi run export``.
"""

from __future__ import annotations

import math

from max import functional as F
from max.driver import Device
from max.dtype import DType
from max.graph import Dim
from max.nn import Linear, Module
from max.nn.kernels.attention import flash_attention
from max.tensor import Tensor
from max.tensor.sharding import DeviceMesh

from minillm.config import LlamaConfig
from minillm.layers import apply_rope, rope_cos_sin


def causal_mask(
    q_len: int | Dim, kv_len: int | Dim, device: Device | DeviceMesh
) -> Tensor:
    """Returns a ``[q_len, kv_len]`` additive mask for queries at the last
    ``q_len`` of ``kv_len`` positions: 0 where a query may attend, -inf
    where it may not. The lengths may be ints or symbolic dimensions."""
    blocked = F.broadcast_to(
        Tensor(float("-inf"), dtype=DType.float32, device=device),
        [q_len, kv_len],
    )
    return F.band_part(
        blocked, num_lower=None, num_upper=kv_len - q_len, exclude=True
    )


def repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    """Repeats each head of ``[batch, seq, kv_heads, dim]`` ``n_rep`` times."""
    batch, seq, kv_heads, dim = x.shape
    x = x.reshape([batch, seq, kv_heads, 1, dim])
    x = F.broadcast_to(x, [batch, seq, kv_heads, n_rep, dim])
    return x.reshape([batch, seq, kv_heads * n_rep, dim])


def naive_attention(q: Tensor, k: Tensor, v: Tensor, mask: Tensor) -> Tensor:
    """Computes causal attention with basic operations.

    ``q`` is ``[batch, seq, heads, dim]``; ``k`` and ``v`` may have fewer
    heads, which are repeated to match.
    """
    k = repeat_kv(k, q.shape[2] // k.shape[2])
    v = repeat_kv(v, q.shape[2] // v.shape[2])
    q, k, v = (t.permute([0, 2, 1, 3]) for t in (q, k, v))  # heads first
    scores = (q @ k.transpose(-1, -2)) / math.sqrt(int(q.shape[-1])) + mask
    return (F.softmax(scores) @ v).permute([0, 2, 1, 3])


class Attention(Module):
    """Causal grouped-query attention with rotary position embeddings."""

    def __init__(self, config: LlamaConfig) -> None:
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.rope_theta = config.rope_theta
        hidden = config.hidden_size
        self.q_proj = Linear(hidden, self.num_heads * self.head_dim, bias=False)
        self.k_proj = Linear(
            hidden, self.num_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = Linear(
            hidden, self.num_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = Linear(self.num_heads * self.head_dim, hidden, bias=False)

    def project(
        self, x: Tensor, start: int | Dim | Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Returns RoPE-rotated queries and keys, and values, for ``x``,
        whose first position is ``start``."""
        batch, seq, _ = x.shape
        q = self.q_proj(x).reshape([batch, seq, self.num_heads, self.head_dim])
        k = self.k_proj(x).reshape(
            [batch, seq, self.num_kv_heads, self.head_dim]
        )
        v = self.v_proj(x).reshape(
            [batch, seq, self.num_kv_heads, self.head_dim]
        )
        cos, sin = rope_cos_sin(
            self.head_dim, self.rope_theta, start, seq, x.mesh
        )
        return apply_rope(q, cos, sin), apply_rope(k, cos, sin), v

    def attend(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """Attends from ``q`` to ``k`` and ``v`` and projects the heads back."""
        batch, q_len = q.shape[0], q.shape[1]
        mask = causal_mask(q_len, k.shape[1], q.mesh)
        mask = F.broadcast_to(mask, [batch, q_len, k.shape[1]])
        out = flash_attention(q, k, v, 1 / math.sqrt(self.head_dim), mask)
        # Attention never mixes heads, so the output is placed like the
        # queries. Chapter 2 relies on this when heads span several devices.
        out = out.rebind_mapping(q.mapping)
        return self.o_proj(
            out.reshape([batch, q_len, self.num_heads * self.head_dim])
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.attend(*self.project(x, 0))

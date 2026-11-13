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
"""Generated from 01_04_building_blocks.ipynb by tools/nb2book.py.

Edit the notebook, then run ``pixi run export``.
"""

from __future__ import annotations

from max import functional as F
from max.driver import Device
from max.dtype import DType
from max.graph import Dim
from max.nn import Linear, Module
from max.tensor import Tensor
from max.tensor.sharding import DeviceMesh

from minillm.config import LlamaConfig


class MLP(Module):
    """The SwiGLU feed-forward network of a Llama layer."""

    def __init__(self, config: LlamaConfig) -> None:
        hidden, inner = config.hidden_size, config.intermediate_size
        self.gate_proj = Linear(hidden, inner, bias=False)
        self.up_proj = Linear(hidden, inner, bias=False)
        self.down_proj = Linear(inner, hidden, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def rope_cos_sin(
    head_dim: int,
    theta: float,
    start: int | Dim | Tensor,
    length: int | Dim,
    device: Device | DeviceMesh,
) -> tuple[Tensor, Tensor]:
    """Returns the RoPE cos and sin tables, ``[length, head_dim]``, for
    positions ``start`` to ``start + length - 1``."""
    inv_freq = Tensor(
        [theta ** (-i / head_dim) for i in range(0, head_dim, 2)],
        dtype=DType.float32,
        device=device,
    )
    positions = (
        Tensor.arange(length, dtype=DType.float32, device=device) + start
    )
    angles = F.outer(positions, inv_freq)
    angles = F.concat([angles, angles], axis=-1)
    return F.cos(angles), F.sin(angles)


def rotate_half(x: Tensor) -> Tensor:
    half = x.shape[-1] // 2
    return F.concat([-x[..., half:], x[..., :half]], axis=-1)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Rotates ``x``, shaped ``[batch, seq, heads, head_dim]``."""
    seq, dim = cos.shape
    cos = cos.reshape([1, seq, 1, dim])
    sin = sin.reshape([1, seq, 1, dim])
    return x * cos + rotate_half(x) * sin

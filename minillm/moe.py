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
"""Generated from 06_03_mixture_of_experts.ipynb by tools/nb2book.py.

Edit the notebook, then run ``pixi run export``.
"""

from __future__ import annotations

from max import functional as F
from max.dtype import DType
from max.nn import Linear, Module
from max.tensor import Tensor


class Experts(Module):
    """The stacked weights of every expert: gate and up projections fused."""

    def __init__(
        self, num_experts: int, hidden: int, intermediate: int
    ) -> None:
        self.gate_up_proj = Tensor.zeros(
            [num_experts, 2 * intermediate, hidden]
        )
        self.down_proj = Tensor.zeros([num_experts, hidden, intermediate])


class MoE(Module):
    """A mixture-of-experts layer with top-k routing, as in Qwen3-MoE."""

    def __init__(
        self, num_experts: int, top_k: int, hidden: int, intermediate: int
    ) -> None:
        self.num_experts, self.top_k, self.intermediate = (
            num_experts,
            top_k,
            intermediate,
        )
        self.gate = Linear(hidden, num_experts, bias=False)  # the router
        self.experts = Experts(num_experts, hidden, intermediate)

    def routing(self, x: Tensor) -> Tensor:
        """Returns each token's weight for each expert, ``[tokens, experts]``,
        zero for experts the token wasn't routed to."""
        tokens = x.shape[0]
        weights, chosen = F.top_k(F.softmax(self.gate(x)), self.top_k)
        weights = weights / weights.sum(axis=-1)
        ids = Tensor.arange(self.num_experts, dtype=DType.int64, device=x.mesh)
        one_hot = (
            chosen.reshape([tokens, self.top_k, 1])
            == ids.reshape([1, 1, self.num_experts])
        ).cast(x.dtype)
        return (
            (one_hot * weights.reshape([tokens, self.top_k, 1]))
            .sum(axis=1)
            .reshape([tokens, self.num_experts])
        )

    def forward(self, x: Tensor) -> Tensor:
        """``x`` is ``[tokens, hidden]``."""
        tokens, hidden = x.shape
        gate_up = x.reshape(
            [1, tokens, hidden]
        ) @ self.experts.gate_up_proj.transpose(1, 2)
        gate, up = (
            gate_up[:, :, : self.intermediate],
            gate_up[:, :, self.intermediate :],
        )
        outputs = (F.silu(gate) * up) @ self.experts.down_proj.transpose(
            1, 2
        )  # [E, T, H]
        weights = self.routing(x).T.reshape([self.num_experts, tokens, 1])
        return (weights * outputs).sum(axis=0).reshape([tokens, hidden])

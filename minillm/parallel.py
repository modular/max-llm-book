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
"""Generated from 02_02_tensor_parallel.ipynb by tools/nb2book.py.

Edit the notebook, then run ``pixi run export``.
"""

from __future__ import annotations

from max.nn.common_layers.linear import col_parallel, row_parallel

from minillm.model import LlamaForCausalLM


def tensor_parallel(
    model: LlamaForCausalLM, axis: str = "tp"
) -> LlamaForCausalLM:
    """Shards every layer's projections across mesh axis ``axis``."""
    for layer in model.model.layers:
        attn, mlp = layer.self_attn, layer.mlp
        for linear in (
            attn.q_proj,
            attn.k_proj,
            attn.v_proj,
            mlp.gate_proj,
            mlp.up_proj,
        ):
            col_parallel(linear, axis)
        for linear in (attn.o_proj, mlp.down_proj):
            row_parallel(linear, axis)
    return model

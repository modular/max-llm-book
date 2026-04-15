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
# gpt2_arch/weight_adapters.py
"""Weight adapter for GPT-2 HuggingFace safetensors.

GPT-2 uses Conv1D (not Linear) for attention and MLP projections.
Conv1D stores weights as [in_features, out_features]; MAX Linear expects
[out_features, in_features]. This adapter transposes the affected layers.

Also handles:
- Key remapping: HF safetensors use ``h.0.ln_1.weight``; our model expects
  ``transformer.h.0.ln_1.weight`` (the ``transformer.`` prefix).
- The tied embedding: GPT-2 small doesn't include ``lm_head.weight``
  in the safetensors file because it's tied to ``transformer.wte.weight``.
- Skipping the causal attention mask buffers (``h.*.attn.bias`` and
  ``h.*.attn.masked_bias``) which are not trainable parameters.
"""

# ANCHOR: book
from __future__ import annotations

import numpy as np
from max.graph.weights import WeightData, Weights

# Layer name suffixes that use Conv1D and need transposing
_CONV1D_LAYERS = ("c_attn", "c_proj", "c_fc")

# Keys in the safetensors that are causal-mask buffers, not parameters.
_SKIP_SUFFIXES = (".attn.bias", ".attn.masked_bias")


def _to_numpy(wd: WeightData) -> np.ndarray:
    # np.from_dlpack() reads via DLPack; np.array() then copies into new,
    # contiguous, writable memory — required by compile().
    return np.array(np.from_dlpack(wd))


def convert_safetensor_state_dict(
    state_dict: dict[str, Weights],
    **unused_kwargs,
) -> dict[str, WeightData]:
    result: dict[str, WeightData] = {}

    for key, value in state_dict.items():
        # Skip causal-mask buffers — they are not model parameters.
        if any(key.endswith(suffix) for suffix in _SKIP_SUFFIXES):
            continue

        mapped_key = (
            key if key.startswith("transformer.") else f"transformer.{key}"
        )
        arr = _to_numpy(value.data())

        # Conv1D stores [in, out]; MAX Linear expects [out, in].
        if any(
            layer in mapped_key for layer in _CONV1D_LAYERS
        ) and mapped_key.endswith(".weight"):
            arr = np.ascontiguousarray(arr.T)

        result[mapped_key] = WeightData.from_numpy(arr, mapped_key)

    # GPT-2 small: lm_head weight is tied to wte; add it explicitly.
    wte_key = "transformer.wte.weight"
    if "lm_head.weight" not in result and wte_key in result:
        wte_arr = np.array(result[wte_key].data)
        result["lm_head.weight"] = WeightData.from_numpy(
            wte_arr, "lm_head.weight"
        )

    return result


# ANCHOR_END: book

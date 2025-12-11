# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from __future__ import annotations

import numpy as np
from max.graph.weights import WeightData, Weights

# Mapping from HuggingFace GPT-2 safetensor names to MAX format
# Note: GPT-2 safetensors don't have 'transformer.' prefix
GPT2_SAFETENSOR_MAP: dict[str, str] = {
    "wte.": "language_model.wte.",
    "wpe.": "language_model.wpe.",
    "ln_f.": "language_model.ln_f.",
    "h.": "language_model.h.",
}

# Weights that need to be transposed (Conv1D -> Linear)
# GPT-2 uses Conv1D which stores weights as [in_features, out_features]
# Linear expects [out_features, in_features]
TRANSPOSE_WEIGHTS = [
    ".c_attn.weight",
    ".c_proj.weight",
    ".c_fc.weight",
]


def convert_safetensor_state_dict(
    state_dict: dict[str, Weights], **kwargs
) -> dict[str, WeightData]:
    """Convert safetensor state dict to MAX format.

    Args:
        state_dict: Dictionary of weight tensors

    Returns:
        Dictionary of converted weight data (raw arrays, not WeightData objects)

    Note:
        Despite the return type hint, this function returns raw numpy arrays,
        not WeightData objects. This follows the pattern used in gpt_oss_module_v3.
    """
    new_state_dict: dict[str, WeightData] = {}
    wte_array = None  # Keep track of wte array for tying

    for weight_name, value in state_dict.items():
        max_name: str = weight_name

        # Skip attention bias buffers (causal masks) - we generate these dynamically
        if weight_name.endswith(".attn.bias"):
            continue

        # Remap weight names from HuggingFace to MAX format
        for before, after in GPT2_SAFETENSOR_MAP.items():
            max_name = max_name.replace(before, after)

        # Get the weight data and convert to numpy array
        weight_data = value.data()
        arr = np.array(np.from_dlpack(weight_data), copy=True)

        # Transpose Conv1D weights to Linear format
        needs_transpose = any(pat in weight_name for pat in TRANSPOSE_WEIGHTS)
        if needs_transpose:
            # Conv1D: [in_features, out_features] -> Linear: [out_features, in_features]
            arr = np.ascontiguousarray(arr.T)
        else:
            # Ensure all arrays are contiguous
            arr = np.ascontiguousarray(arr)

        # Keep wte array for tying embeddings
        if max_name == "language_model.wte.weight":
            wte_array = arr

        # Return raw array (like gpt_oss_module_v3), not WeightData object
        new_state_dict[max_name] = arr

    # Handle tied embeddings - if lm_head.weight is missing, copy from wte.weight
    if "language_model.lm_head.weight" not in new_state_dict:
        if wte_array is not None:
            new_state_dict["language_model.lm_head.weight"] = wte_array.copy()

    return new_state_dict

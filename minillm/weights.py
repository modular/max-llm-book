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
"""Generated from 01_03_tokens_and_weights.ipynb, 01_04_building_blocks.ipynb by tools/nb2book.py.

Edit the notebook, then run ``pixi run export``.
"""

from __future__ import annotations

import json

import numpy as np
from huggingface_hub import hf_hub_download

from minillm.config import MODEL_ID

_NUMPY_DTYPES = {
    "F32": np.float32,
    "F16": np.float16,
    "I64": np.int64,
    "I32": np.int32,
}


def read_safetensors(path: str) -> dict[str, np.ndarray]:
    """Reads a safetensors file, converting floating-point tensors to float32."""
    with open(path, "rb") as f:
        header_len = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(header_len))
    data = np.memmap(path, dtype=np.uint8, mode="r", offset=8 + header_len)
    tensors = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        start, end = info["data_offsets"]
        raw = data[start:end]
        if info["dtype"] == "BF16":
            array = (raw.view(np.uint16).astype(np.uint32) << 16).view(
                np.float32
            )
        else:
            array = raw.view(_NUMPY_DTYPES[info["dtype"]])
            if array.dtype == np.float16:
                array = array.astype(np.float32)
        tensors[name] = array.reshape(info["shape"])
    return tensors


def load_checkpoint(model_id: str = MODEL_ID) -> dict[str, np.ndarray]:
    """Downloads a model's safetensors checkpoint and returns its tensors."""
    return read_safetensors(hf_hub_download(model_id, "model.safetensors"))


def select(
    weights: dict[str, np.ndarray], prefix: str
) -> dict[str, np.ndarray]:
    """Returns the entries under ``prefix``, with the prefix removed."""
    return {
        name.removeprefix(prefix): value
        for name, value in weights.items()
        if name.startswith(prefix)
    }

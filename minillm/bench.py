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
"""Generated from 04_03_kernel_benchmarks.ipynb by tools/nb2book.py.

Edit the notebook, then run ``pixi run export``.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass
class Timing:
    """Milliseconds per call over the measured iterations."""

    median_ms: float
    p90_ms: float
    min_ms: float

    def __str__(self) -> str:
        return f"median {self.median_ms:7.2f} ms  p90 {self.p90_ms:7.2f} ms"


def benchmark(
    fn: Callable[[], object], warmup: int = 3, iterations: int = 20
) -> Timing:
    """Times ``fn``, which must return a fully computed result (for example,
    a NumPy array read back from the device)."""
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000)
    return Timing(
        median_ms=float(np.median(samples)),
        p90_ms=float(np.percentile(samples, 90)),
        min_ms=float(np.min(samples)),
    )

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
"""Generated from 04_01_layer_tests.ipynb by tools/nb2book.py.

Edit the notebook, then run ``pixi run export``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CorrectnessResult:
    """How far an output is from its reference, as MAX's layer tests report it."""

    label: str
    mae: float
    max_abs_err: float
    cos_dist: float
    passed: bool

    def __str__(self) -> str:
        status = "pass" if self.passed else "FAIL"
        return (
            f"{self.label:16} mae {self.mae:.1e}  max {self.max_abs_err:.1e}  "
            f"cos dist {self.cos_dist:.1e}  {status}"
        )


def compare_outputs(
    actual: np.ndarray,
    expected: np.ndarray,
    label: str = "",
    atol: float = 1e-3,
    rtol: float = 1e-3,
    cos_threshold: float = 1e-5,
) -> CorrectnessResult:
    """Compares ``actual`` with ``expected``; passes when they are allclose
    and their cosine distance is below ``cos_threshold``."""
    a = actual.astype(np.float64).ravel()
    e = expected.astype(np.float64).ravel()
    cos_dist = 1.0 - float(
        a @ e / (np.linalg.norm(a) * np.linalg.norm(e) + 1e-12)
    )
    allclose = bool(np.allclose(actual, expected, atol=atol, rtol=rtol))
    return CorrectnessResult(
        label=label,
        mae=float(np.mean(np.abs(a - e))),
        max_abs_err=float(np.max(np.abs(a - e))),
        cos_dist=cos_dist,
        passed=allclose and cos_dist < cos_threshold,
    )

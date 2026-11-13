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
"""Runs every section notebook top to bottom with papermill.

Each notebook checks its own results with ``assert`` statements: logits
against references, tokens against other implementations, and so on. A
failed check or any other error fails that notebook's test. The notebooks
run on MAX's default device, a GPU if there is one, and download models
from Hugging Face, so these tests need the network and take a while. Run
them with ``pixi run test-notebooks``.
"""

from __future__ import annotations

from pathlib import Path

import papermill
import pytest
from huggingface_hub import get_token

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = sorted((ROOT / "notebooks").glob("[0-9][0-9]_*.ipynb"))
# Notebooks whose models are gated on Hugging Face: they need a token for an
# account that accepted the model's license, so they're skipped without one.
GATED = {"06_02_gemma3": "google/gemma-3-270m-it"}
# Two quick notebooks that cover the eager APIs, loading SmolLM2's weights,
# and building and calling a Mojo custom op. Nightly CI runs only these
# (`pixi run test-notebooks-smoke`); the unit tests cover the model itself on
# a tiny random checkpoint.
SMOKE = {"01_02_tensors_and_eager", "03_01_custom_ops"}


@pytest.mark.notebooks
@pytest.mark.parametrize(
    "notebook",
    [
        pytest.param(nb, marks=pytest.mark.smoke) if nb.stem in SMOKE else nb
        for nb in NOTEBOOKS
    ],
    ids=lambda p: p.stem,
)
def test_notebook_runs_and_checks_pass(notebook: Path, tmp_path: Path) -> None:
    if notebook.stem in GATED and get_token() is None:
        pytest.skip(f"{GATED[notebook.stem]} is gated; set HF_TOKEN to run it")
    papermill.execute_notebook(
        str(notebook),
        str(tmp_path / notebook.name),
        cwd=str(notebook.parent),
        kernel_name="python3",
        progress_bar=False,
        execution_timeout=1800,
    )

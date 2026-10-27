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
"""Smoke tests for notebooks/tutorial.ipynb.

These tests validate the notebook *structure* without executing it:
- Every code cell is syntactically valid Python (ast.parse).
- The notebook section headings include all chapters listed in SUMMARY.md.

Full execution (pixi run notebook → Run All Cells) is left to the developer
because it requires a ~500 MB weight download and compilation time that is
inappropriate for a pre-submit test.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import nbformat
import pytest

REPO_ROOT = Path(__file__).parent.parent
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "tutorial.ipynb"
SUMMARY_PATH = REPO_ROOT / "book" / "src" / "SUMMARY.md"

# Steps that must appear as headings in the notebook (derived from SUMMARY.md).
# The notebook uses "## Step NN" headings; verify the key ones are present.
REQUIRED_STEP_PREFIXES = [f"Step {i:02d}" for i in range(1, 13)]


@pytest.fixture(scope="module")
def notebook() -> nbformat.NotebookNode:
    with NOTEBOOK_PATH.open() as f:
        return nbformat.read(f, as_version=4)


def _code_cells(nb: nbformat.NotebookNode) -> list[nbformat.NotebookNode]:
    return [c for c in nb.cells if c.cell_type == "code"]


def _markdown_cells(nb: nbformat.NotebookNode) -> list[nbformat.NotebookNode]:
    return [c for c in nb.cells if c.cell_type == "markdown"]


def _all_markdown_text(nb: nbformat.NotebookNode) -> str:
    return "\n".join(c.source for c in _markdown_cells(nb))


class TestNotebookStructure:
    def test_notebook_exists(self) -> None:
        assert NOTEBOOK_PATH.exists(), f"Notebook not found: {NOTEBOOK_PATH}"

    def test_notebook_is_valid_nbformat(
        self, notebook: nbformat.NotebookNode
    ) -> None:
        assert notebook.nbformat == 4
        assert len(notebook.cells) > 0

    def test_has_code_cells(self, notebook: nbformat.NotebookNode) -> None:
        code_cells = _code_cells(notebook)
        assert len(code_cells) >= 10, (
            f"Expected at least 10 code cells, found {len(code_cells)}"
        )

    def test_has_markdown_cells(self, notebook: nbformat.NotebookNode) -> None:
        md_cells = _markdown_cells(notebook)
        assert len(md_cells) >= 12, (
            f"Expected at least 12 markdown cells, found {len(md_cells)}"
        )


class TestCodeCellSyntax:
    def test_all_code_cells_are_valid_python(
        self, notebook: nbformat.NotebookNode
    ) -> None:
        errors: list[str] = []
        for i, cell in enumerate(_code_cells(notebook)):
            cell_id = cell.get("id", f"index-{i}")
            source = cell.source
            if not source.strip():
                continue  # skip empty cells
            try:
                ast.parse(source)
            except SyntaxError as exc:
                errors.append(f"Cell '{cell_id}' has a syntax error: {exc}")

        assert not errors, "\n".join(errors)

    def test_setup_cell_imports_max(
        self, notebook: nbformat.NotebookNode
    ) -> None:
        code_cells = _code_cells(notebook)
        assert code_cells, "No code cells found"
        first_code = code_cells[0].source
        assert "max.functional" in first_code, (
            "Setup cell should import max.functional"
        )
        assert "GPT2Config" in first_code, (
            "Setup cell should import GPT2Config from gpt2_arch.gpt2"
        )

    def test_generation_cell_present(
        self, notebook: nbformat.NotebookNode
    ) -> None:
        all_code = "\n".join(c.source for c in _code_cells(notebook))
        assert (
            "greedy_next_token" in all_code or "compiled_model" in all_code
        ), "Notebook should contain a generation cell using the compiled model"


class TestNotebookSections:
    def test_all_steps_covered(self, notebook: nbformat.NotebookNode) -> None:
        md_text = _all_markdown_text(notebook)
        missing = [
            prefix for prefix in REQUIRED_STEP_PREFIXES if prefix not in md_text
        ]
        assert not missing, (
            f"Notebook is missing sections for: {missing}\n"
            f"Each of Step 01 through Step 12 must appear in a markdown heading."
        )

    def test_summary_steps_match_notebook(self) -> None:
        """Chapters in SUMMARY.md should be reflected in notebook headings."""
        summary_text = SUMMARY_PATH.read_text()
        # Extract step file references from SUMMARY: step_01.md ... step_12.md
        step_refs = re.findall(r"step_(\d+)\.md", summary_text)
        assert step_refs, "Could not find step_XX.md references in SUMMARY.md"

        nb = nbformat.read(NOTEBOOK_PATH.open(), as_version=4)
        md_text = _all_markdown_text(nb)

        missing = []
        for ref in step_refs:
            step_num = int(ref)
            prefix = f"Step {step_num:02d}"
            if prefix not in md_text:
                missing.append(prefix)

        assert not missing, (
            f"SUMMARY.md references {missing} but those headings are absent from the notebook."
        )

    def test_weight_loading_section_present(
        self, notebook: nbformat.NotebookNode
    ) -> None:
        md_text = _all_markdown_text(notebook)
        assert "pretrained" in md_text.lower(), (
            "Notebook should have a section about loading pretrained weights"
        )

    def test_generation_section_present(
        self, notebook: nbformat.NotebookNode
    ) -> None:
        md_text = _all_markdown_text(notebook)
        assert (
            "generation" in md_text.lower() or "generate" in md_text.lower()
        ), "Notebook should have a text generation section"

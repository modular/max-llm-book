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
"""Checks that files derived from the notebooks are up to date.

The notebooks are the source of truth. If a check here fails, run
``pixi run bootstrap`` or ``pixi run export`` and commit the result.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import nbformat
import pytest
import tomllib

ROOT = Path(__file__).resolve().parent.parent
NB2BOOK = ROOT / "tools" / "nb2book.py"
NOTEBOOKS = sorted((ROOT / "notebooks").glob("[0-9][0-9]_*.ipynb"))

# Every page of handbook.modular.com, from its sitemap.
HANDBOOK_PAGES = {
    "",
    "llm-inference-basics/",
    "llm-inference-basics/what-is-llm-inference/",
    "llm-inference-basics/training-inference-differences/",
    "llm-inference-basics/how-does-llm-inference-work/",
    "llm-inference-basics/cpu-vs-gpu-vs-tpu/",
    "llm-inference-basics/llm-inference-metrics/",
    "getting-started/",
    "getting-started/serverless-vs-self-hosted-llm-inference/",
    "getting-started/choosing-the-right-model/",
    "getting-started/choosing-the-right-gpu/",
    "getting-started/calculating-gpu-memory-for-llms/",
    "getting-started/choosing-the-right-inference-framework/",
    "getting-started/bring-your-own-cloud/",
    "getting-started/on-prem-llms/",
    "model-preparation/",
    "model-preparation/llm-fine-tuning/",
    "model-preparation/llm-quantization/",
    "model-preparation/llm-distillation/",
    "model-interaction/",
    "model-interaction/openai-compatible-api/",
    "model-interaction/anthropic-compatible-api/",
    "model-interaction/function-calling/",
    "model-interaction/structured-outputs/",
    "model-interaction/model-context-protocol/",
    "model-interaction/prompt-engineering/",
    "model-interaction/agent-harnesses/",
    "model-interaction/inference-parameters/",
    "inference-optimization/",
    "inference-optimization/llm-performance-benchmarks/",
    "inference-optimization/static-dynamic-continuous-batching/",
    "inference-optimization/pagedattention/",
    "inference-optimization/speculative-decoding/",
    "inference-optimization/prefill-decode-disaggregation/",
    "inference-optimization/prefix-caching/",
    "inference-optimization/inference-routing/",
    "inference-optimization/kv-cache-offloading/",
    "inference-optimization/data-tensor-pipeline-expert-hybrid-parallelism/",
    "inference-optimization/offline-batch-inference/",
    "kernel-optimization/",
    "kernel-optimization/kernel-optimization-for-llm-inference/",
    "kernel-optimization/gpu-architecture-fundamentals/",
    "kernel-optimization/gpu-architecture-fundamentals/gpu-memory/",
    "kernel-optimization/gpu-architecture-fundamentals/streaming-multiprocessors/",
    "kernel-optimization/gpu-architecture-fundamentals/tensor-cores/",
    "kernel-optimization/gpu-architecture-fundamentals/threads-warps-blocks/",
    "kernel-optimization/kernel-optimization-tools/",
    "kernel-optimization/flashattention/",
    "infrastructure-and-operations/",
    "infrastructure-and-operations/what-is-llm-inference-infrastructure/",
    "infrastructure-and-operations/build-and-maintenance-cost/",
    "infrastructure-and-operations/distributed-inference/",
    "infrastructure-and-operations/comprehensive-observability/",
    "infrastructure-and-operations/fast-scaling/",
    "infrastructure-and-operations/multi-model-inference-pipelines/",
    "infrastructure-and-operations/multi-cloud-and-cross-region-inference/",
    "infrastructure-and-operations/inferenceops-and-management/",
}
HANDBOOK_LINK = re.compile(r"https://handbook\.modular\.com/([^\s)\"'>#]*)")


def run(command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(NB2BOOK), command, "--check"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )


@pytest.mark.parametrize("command", ["bootstrap", "export"])
def test_generated_files_match_notebooks(command: str) -> None:
    result = run(command)
    assert result.returncode == 0, (
        f"Run `pixi run {command}` and commit the result.\n{result.stderr}"
    )


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_handbook_links_name_real_pages(path: Path) -> None:
    text = path.read_text()
    for page in HANDBOOK_LINK.findall(text):
        assert page in HANDBOOK_PAGES, (
            f"{path.name} links https://handbook.modular.com/{page}, which is "
            "not a known handbook page (pages end with a slash)."
        )


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebooks_are_committed_without_outputs(path: Path) -> None:
    nb = nbformat.read(path, as_version=4)
    for index, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            assert not cell.outputs, (
                f"{path.name} cell {index} has outputs; clear them before "
                "committing (the book pages carry the outputs)."
            )


LOCAL_PATH_RE = re.compile(r"/Users/|/home/\w|/var/folders/|ipykernel_\d")


@pytest.mark.parametrize(
    "path",
    sorted((ROOT / "book" / "src").glob("*.md")),
    ids=lambda p: p.name,
)
def test_pages_name_no_local_paths(path: Path) -> None:
    for number, line in enumerate(path.read_text().splitlines(), 1):
        assert not LOCAL_PATH_RE.search(line), (
            f"{path.name}:{number} names a path on the machine that rendered "
            f"it: {line.strip()}"
        )


DIAGRAM_URL_RE = re.compile(
    r"https://llm\.modular\.com/assets/diagrams/([\w-]+)\.svg"
)


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_diagrams_have_light_and_dark_svgs(path: Path) -> None:
    svgs = ROOT / "book" / "src" / "assets" / "diagrams"
    for name in DIAGRAM_URL_RE.findall(path.read_text()):
        for svg in (f"{name}.svg", f"{name}-dark.svg"):
            assert (svgs / svg).exists(), (
                f"{path.name} shows diagram {name!r}, but {svg} is missing; "
                "run `python tools/diagrams.py`."
            )


def test_outline_lists_every_notebook() -> None:
    outline = tomllib.loads((ROOT / "book" / "outline.toml").read_text())
    listed = {
        section.partition("|")[0]
        for chapter in outline["chapter"]
        for section in chapter["sections"]
    }
    missing = [p.stem for p in NOTEBOOKS if p.stem not in listed]
    assert not missing, f"add these notebooks to book/outline.toml: {missing}"


# Colors that are the same in both themes, so the page script leaves them.
SAME_IN_DARK = {"#008300"}


def test_page_script_has_a_dark_step_for_every_figure_color() -> None:
    viz_colors = set(
        re.findall(r"#[0-9a-f]{6}", (ROOT / "minillm" / "viz.py").read_text())
    )
    script = (ROOT / "book" / "theme" / "plotly-figures.js").read_text()
    dark_map = set(re.findall(r'"(#[0-9a-f]{6})":', script))
    missing = viz_colors - dark_map - SAME_IN_DARK
    assert not missing, (
        f"add dark-mode steps for {sorted(missing)} to book/theme/plotly-figures.js"
    )
    uppercase = re.findall(
        r"#[0-9A-F]{6}", (ROOT / "minillm" / "viz.py").read_text()
    )
    assert not [c for c in uppercase if c != c.lower()], (
        "write colors in lowercase"
    )


def test_links_between_pages_resolve() -> None:
    pages = {path.stem for path in (ROOT / "book" / "src").glob("*.md")}
    broken = []
    for path in NOTEBOOKS:
        for cell in nbformat.read(path, as_version=4).cells:
            if cell.cell_type != "markdown":
                continue
            for page in re.findall(
                r"https://llm\.modular\.com/([\w-]+)\.html", cell.source
            ):
                if page not in pages:
                    broken.append(f"{path.name}: {page}")
    for path in (ROOT / "book" / "src").glob("*.md"):
        for page in re.findall(
            r"\]\(\.?/?([\w-]+)\.md(?:#[\w-]+)?\)", path.read_text()
        ):
            if page not in pages:
                broken.append(f"{path.name}: {page}")
    assert not broken, broken

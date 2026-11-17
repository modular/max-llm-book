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
"""Builds the book and the ``minillm`` package from the chapter notebooks.

The notebooks in ``notebooks/`` are the single source of truth. This tool
derives everything else from them:

``bootstrap``
    Copies ``notebooks/_bootstrap.py`` into the first cell of every notebook,
    so all chapters install MAX the same way on Colab and Kaggle.
``export``
    Writes each code cell that starts with ``# export: minillm/<file>.py``
    into that file. Later chapters import the package instead of repeating
    the code.
``render``
    Writes one mdBook page per notebook to ``book/src/``, with Colab and
    Kaggle badges, and regenerates the chapter list in ``SUMMARY.md``. With
    ``--execute`` it runs each notebook first and includes its outputs.
    Without it, a notebook whose code cells are unchanged since its last
    execution takes its outputs from that run.
    Cell tags control the page: ``hide-in-book`` drops a cell,
    ``hide-input`` keeps only its output, and ``hide-output`` keeps only its
    code. A code cell longer than ``FOLD_LINES`` renders folded behind a
    summary that the reader opens; ``collapse-input`` folds a shorter cell,
    and ``expand-input`` keeps a long one open.

Each command takes ``--check``, which fails instead of writing when a file
would change.
"""

from __future__ import annotations

import argparse
import ast
import atexit
import base64
import copy
import html
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterable
from pathlib import Path

import nbformat
import tomllib
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = ROOT / "notebooks"
BOOK_SRC = ROOT / "book" / "src"
GENERATED_IMAGES = BOOK_SRC / "generated"
PLOTLY_MIME = "application/vnd.plotly.v1+json"
SUMMARY = BOOK_SRC / "SUMMARY.md"
OUTLINE = ROOT / "book" / "outline.toml"
# The last executed copy of each notebook, kept out of git, so that a prose
# edit can be rendered without running the notebook again.
EXECUTED = ROOT / "book" / ".executed"
BOOTSTRAP = NOTEBOOKS / "_bootstrap.py"

GITHUB_REPO = "modular/max-llm-book"
GITHUB_BRANCH = "main"

EXPORT_RE = re.compile(r"^# export: (minillm/[\w/]+\.py)\s*$")
# Notebooks link other sections by their published URL, which works in Colab
# and Kaggle too. In the book the same link becomes a relative one, so it
# stays on the copy being read, such as a local preview.
BOOK_URL = "https://llm.modular.com/"
BOOK_LINK_RE = re.compile(
    r"\]\(https://llm\.modular\.com/([\w-]+)\.html(#[\w-]+)?\)"
)
# Notebooks show a diagram from tools/diagrams.py by its published light SVG,
# which works in Colab and Kaggle too. The book shows a light and a dark
# copy, and the theme's CSS hides the one that doesn't match.
DIAGRAM_RE = re.compile(
    r"!\[([^\]]*)\]\(https://llm\.modular\.com/assets/diagrams/([\w-]+)\.svg\)"
)
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
# Progress bars, download chatter and MAX's tuning hints are noise on a
# static page.
NOISY_STDERR_RE = re.compile(
    r"it/s\]|s/it\]|B/s\]|Downloading|Fetching \d+ files|reconstructing file|"
    r"downloading bytes|^WARNING:max\.|^INFO:max\."
)
MAX_OUTPUT_LINES = 40
# Warnings and tracebacks name source files by absolute path, which would
# publish the author's home directory. Pages show repository files relative
# to the repository and notebook cells as <cell>.
# The lookbehind skips URLs, such as the GitHub links in the badges.
LOCAL_PATH_SUBS = (
    (re.compile(r"(?<![\w.:/])/\S*?/max-llm-book/"), ""),
    (re.compile(r"(?<![\w.:/])/\S*?/ipykernel_\d+/\d+\.py"), "<cell>"),
    (re.compile(r"(?<![\w.:/])/(?:Users|home)/[^/\s]+/"), "~/"),
)
# Longer code cells start folded: most readers want the idea and the output,
# and open the code when they want the detail.
FOLD_LINES = 25
MOJO_DEF_RE = re.compile(r"^(?:def|struct|trait)\s+(\w+)", re.MULTILINE)

LICENSE_HEADER = """\
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
"""

SUMMARY_START = "<!-- chapters:start -->"
SUMMARY_END = "<!-- chapters:end -->"


def chapter_notebooks() -> list[Path]:
    """Returns the section notebooks, named ``NN_SS_slug.ipynb`` for section
    SS of chapter NN, in reading order."""
    return sorted(NOTEBOOKS.glob("[0-9][0-9]_*.ipynb"))


def tags(cell: nbformat.NotebookNode) -> set[str]:
    return set(cell.get("metadata", {}).get("tags", []))


def title(nb: nbformat.NotebookNode, path: Path) -> str:
    for cell in nb.cells:
        if cell.cell_type == "markdown":
            for line in cell.source.splitlines():
                if line.startswith("# "):
                    return line[2:].strip()
    raise ValueError(f"{path.name}: no '# Title' heading in a markdown cell")


def write_or_check(path: Path, text: str, check: bool) -> bool:
    """Writes ``text`` to ``path``; returns whether the file changed."""
    old = path.read_text() if path.exists() else None
    if old == text:
        return False
    if check:
        print(f"out of date: {path.relative_to(ROOT)}", file=sys.stderr)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        print(f"wrote {path.relative_to(ROOT)}")
    return True


# --------------------------------------------------------------------------- #
# bootstrap
# --------------------------------------------------------------------------- #


def bootstrap(check: bool) -> bool:
    source = BOOTSTRAP.read_text().split("# --- cell ---\n", 1)[1].rstrip()
    changed = False
    for path in chapter_notebooks():
        nb = nbformat.read(path, as_version=4)
        first = nb.cells[0] if nb.cells else None
        if first is not None and "bootstrap" in tags(first):
            if first.source == source:
                continue
            first.source = source
        else:
            cell = nbformat.v4.new_code_cell(source)
            cell.metadata["tags"] = ["bootstrap"]
            nb.cells.insert(0, cell)
        changed |= write_or_check(path, notebook_text(nb), check)
    return changed


def notebook_text(nb: nbformat.NotebookNode) -> str:
    return nbformat.writes(nb) + "\n"


# --------------------------------------------------------------------------- #
# export
# --------------------------------------------------------------------------- #


def exported_files() -> dict[Path, list[tuple[Path, str]]]:
    """Maps each package file to the (notebook, code) cells exported to it."""
    files: dict[Path, list[tuple[Path, str]]] = {}
    for path in chapter_notebooks():
        nb = nbformat.read(path, as_version=4)
        for cell in nb.cells:
            if cell.cell_type != "code":
                continue
            first, _, body = cell.source.partition("\n")
            match = EXPORT_RE.match(first)
            if match:
                target = ROOT / match.group(1)
                files.setdefault(target, []).append((path, body.strip()))
    return files


def ruff_format(text: str) -> str:
    ruff = shutil.which("ruff")
    if ruff is None:
        return text
    result = subprocess.run(
        [ruff, "format", "--line-length", "80", "-"],
        input=text,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def export(check: bool) -> bool:
    changed = False
    for target, cells in exported_files().items():
        sources = sorted({p.name for p, _ in cells})
        header = (
            LICENSE_HEADER
            + f'"""Generated from {", ".join(sources)} by tools/nb2book.py.\n\n'
            + 'Edit the notebook, then run ``pixi run export``.\n"""\n\n'
        )
        body = "\n\n\n".join(code for _, code in cells)
        text = ruff_format(header + body + "\n")
        changed |= write_or_check(target, text, check)
    return changed


# --------------------------------------------------------------------------- #
# render
# --------------------------------------------------------------------------- #


# Code fences and inline code, which math conversion must leave alone.
CODE_RE = re.compile(r"```.*?```|`[^`\n]*`", re.DOTALL)
DISPLAY_MATH_RE = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
INLINE_MATH_RE = re.compile(r"(?<![\\$])\$(?!\s)([^$\n]+?)(?<!\s)\$(?!\$)")


def _escape_tex(tex: str) -> str:
    # Markdown would otherwise consume backslashes and read `_` and `*`
    # pairs as emphasis before MathJax sees the TeX.
    return tex.replace("\\", "\\\\").replace("_", "\\_").replace("*", "\\*")


def convert_math(markdown: str) -> str:
    """Rewrites Jupyter's ``$`` math delimiters into the ``\\(`` and ``\\[``
    forms that mdBook's MathJax support recognizes."""
    parts = []
    last = 0
    for code in CODE_RE.finditer(markdown):
        parts.append(_convert_math_text(markdown[last : code.start()]))
        parts.append(code.group(0))
        last = code.end()
    parts.append(_convert_math_text(markdown[last:]))
    return "".join(parts)


def _convert_math_text(text: str) -> str:
    text = DISPLAY_MATH_RE.sub(
        lambda m: f"\\\\[{_escape_tex(m.group(1).strip())}\\\\]", text
    )
    return INLINE_MATH_RE.sub(
        lambda m: f"\\\\({_escape_tex(m.group(1))}\\\\)", text
    )


def badges(path: Path) -> str:
    rel = f"notebooks/{path.name}"
    gh = f"github.com/{GITHUB_REPO}/blob/{GITHUB_BRANCH}/{rel}"
    colab = f"https://colab.research.google.com/{gh.replace('github.com', 'github')}"
    kaggle = f"https://kaggle.com/kernels/welcome?src=https://{gh}"
    return (
        f"[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)]({colab})\n"
        f"[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)]({kaggle})\n"
        f"[![View on GitHub](https://img.shields.io/badge/GitHub-notebook-black?logo=github)](https://{gh})\n"
    )


def execute(nb: nbformat.NotebookNode, timeout: int) -> nbformat.NotebookNode:
    nb = copy.deepcopy(nb)
    NotebookClient(
        nb,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOKS)}},
    ).execute()
    return nb


def scrub_paths(text: str) -> str:
    for pattern, replacement in LOCAL_PATH_SUBS:
        text = pattern.sub(replacement, text)
    return text


def clip(text: str) -> str:
    lines = scrub_paths(ANSI_RE.sub("", text)).rstrip().splitlines()
    if len(lines) > MAX_OUTPUT_LINES:
        keep = MAX_OUTPUT_LINES // 2
        lines = lines[:keep] + ["..."] + lines[-keep:]
    return "\n".join(lines)


def render_outputs(
    cell: nbformat.NotebookNode, stem: str, index: int
) -> Iterable[str]:
    if "hide-output" in tags(cell):
        return
    text: list[str] = []
    for n, out in enumerate(cell.get("outputs", [])):
        kind = out.output_type
        if kind == "stream":
            lines = out.text.splitlines(keepends=True)
            if out.name == "stderr":
                lines = [l for l in lines if not NOISY_STDERR_RE.search(l)]
            text.append("".join(lines))
        elif kind in ("execute_result", "display_data"):
            data = out.get("data", {})
            if PLOTLY_MIME in data:
                if text:
                    yield f"```text\n{clip(''.join(text))}\n```"
                    text = []
                name = f"{stem}_{index}_{n}.json"
                GENERATED_IMAGES.mkdir(parents=True, exist_ok=True)
                (GENERATED_IMAGES / name).write_text(
                    json.dumps(data[PLOTLY_MIME], separators=(",", ":"))
                )
                # book/theme/plotly-figures.js draws it.
                yield (
                    f'<div class="plotly-figure" data-src="generated/{name}">'
                    "</div>"
                )
            elif "image/png" in data:
                if text:
                    yield f"```text\n{clip(''.join(text))}\n```"
                    text = []
                name = f"{stem}_{index}_{n}.png"
                GENERATED_IMAGES.mkdir(parents=True, exist_ok=True)
                (GENERATED_IMAGES / name).write_bytes(
                    base64.b64decode(data["image/png"])
                )
                yield f"![output](generated/{name})"
            elif "text/markdown" in data:
                yield data["text/markdown"]
            elif "text/plain" in data:
                # Progress bars in notebooks arrive as display data.
                if not NOISY_STDERR_RE.search(data["text/plain"]):
                    text.append(data["text/plain"] + "\n")
        elif kind == "error":
            raise RuntimeError(
                f"{stem}: cell {index} raised {out.ename}: {out.evalue}"
            )
    if text and clip("".join(text)):
        yield f"```text\n{clip(''.join(text))}\n```"


def fold_summary(code: str, language: str) -> str:
    """Names what a folded cell holds: its imports, its setup, or the
    top-level functions and classes it defines."""
    if language == "mojo":
        names = MOJO_DEF_RE.findall(code)
    else:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            # Cells with IPython magics aren't plain Python.
            return "the code"
        if all(isinstance(n, (ast.Import, ast.ImportFrom)) for n in tree.body):
            return "the imports"
        names = [
            node.name
            for node in tree.body
            if isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            )
        ]
        if not names and isinstance(tree.body[0], (ast.Import, ast.ImportFrom)):
            return "the setup"
    if not names:
        return "the code"
    shown = ", ".join(f"<code>{html.escape(n)}</code>" for n in names[:4])
    return shown + (", ..." if len(names) > 4 else "")


def code_block(code: str, language: str, cell_tags: set[str]) -> str:
    block = f"```{language}\n{code}\n```"
    lines = code.count("\n") + 1
    folded = "collapse-input" in cell_tags or (
        lines > FOLD_LINES and "expand-input" not in cell_tags
    )
    if not folded:
        return block
    what = fold_summary(code, language)
    # mdBook parses the fenced block between the HTML lines as Markdown only
    # when blank lines separate them.
    return (
        f'<details class="code-fold">\n<summary>Show {what} '
        f'<span class="code-fold-lines">{lines} line{"s" * (lines != 1)}'
        "</span></summary>\n\n"
        f"{block}\n\n</details>"
    )


def diagram_html(match: re.Match[str]) -> str:
    alt, name = html.escape(match.group(1)), match.group(2)
    return (
        f'<img class="diagram diagram-light" src="assets/diagrams/{name}.svg"'
        f' alt="{alt}">'
        f'<img class="diagram diagram-dark" src="assets/diagrams/{name}-dark.svg"'
        f' alt="{alt}">'
    )


def render_page(path: Path, nb: nbformat.NotebookNode) -> str:
    parts: list[str] = []
    for index, cell in enumerate(nb.cells):
        cell_tags = tags(cell)
        if "bootstrap" in cell_tags or "hide-in-book" in cell_tags:
            continue
        if cell.cell_type == "markdown":
            source = convert_math(cell.source.strip())
            source = BOOK_LINK_RE.sub(r"](\1.md\2)", source)
            source = DIAGRAM_RE.sub(diagram_html, source)
            if not parts and source.startswith("# "):
                heading, _, rest = source.partition("\n")
                source = f"{heading}\n\n{badges(path)}\n{rest.strip()}"
            parts.append(source)
        elif cell.cell_type == "code":
            first, _, body = cell.source.partition("\n")
            match = EXPORT_RE.match(first)
            code = body.strip() if match else cell.source.strip()
            if code and "hide-input" not in cell_tags:
                language = "mojo" if code.startswith("%%mojo") else "python"
                parts.append(code_block(code, language, cell_tags))
            if match:
                parts.append(f"<small>Saved to `{match.group(1)}`.</small>")
            parts.extend(render_outputs(cell, path.stem, index))
    return "\n\n".join(parts) + "\n"


def render_summary(entries: list[tuple[str, str]], check: bool) -> bool:
    """Writes the sidebar from ``book/outline.toml``: each chapter's landing
    page, then its sections, linked when written and drafts when planned."""
    titles = {page.removesuffix(".md"): title for title, page in entries}
    outline = tomllib.loads(OUTLINE.read_text())
    listed = set()
    listing = []
    for chapter in outline["chapter"]:
        landing = BOOK_SRC / chapter["page"]
        heading = landing.read_text().splitlines()[0].removeprefix("# ")
        listing.append(f"- [{heading}](./{landing.name})\n")
        for section in chapter["sections"]:
            slug, _, planned_title = section.partition("|")
            listed.add(slug)
            if slug in titles:
                listing.append(f"  - [{titles[slug]}](./{slug}.md)\n")
            else:
                listing.append(f"  - [{planned_title or slug}]()\n")
    unlisted = sorted(set(titles) - listed)
    if unlisted:
        raise ValueError(f"add these notebooks to {OUTLINE.name}: {unlisted}")
    text = SUMMARY.read_text()
    start = text.index(SUMMARY_START) + len(SUMMARY_START)
    end = text.index(SUMMARY_END)
    new = text[:start] + "\n\n" + "".join(listing) + "\n" + text[end:]
    return write_or_check(SUMMARY, new, check)


def render(names: list[str], run: bool, timeout: int, check: bool) -> bool:
    changed = False
    entries = []
    failed = []
    if run:
        # Compiled models are cached per run and deleted after, so rendering
        # every notebook can't fill the disk.
        cache = tempfile.mkdtemp(prefix="max-llm-book-cache-")
        os.environ["MODULAR_MAX_CACHE_DIR"] = cache
        atexit.register(shutil.rmtree, cache, ignore_errors=True)
    for path in chapter_notebooks():
        nb = nbformat.read(path, as_version=4)
        page = f"{path.stem}.md"
        entries.append((title(nb, path), page))
        if names and path.stem not in names and path.name not in names:
            continue
        if run:
            print(f"executing {path.name}", flush=True)
            try:
                nb = execute(nb, timeout)
            except CellExecutionError as error:
                # Keep the old page and go on, so one failure doesn't hide others.
                print(f"FAILED {path.name}:\n{str(error)[-2000:]}", flush=True)
                failed.append(path.name)
                continue
            # Output files are named by cell index, so an edit that moves
            # cells would leave the old ones behind.
            if not check:
                for stale in GENERATED_IMAGES.glob(f"{path.stem}_*"):
                    stale.unlink()
                EXECUTED.mkdir(exist_ok=True)
                nbformat.write(nb, EXECUTED / path.name)
        elif (EXECUTED / path.name).exists():
            nb = with_saved_outputs(
                nb, nbformat.read(EXECUTED / path.name, as_version=4)
            )
            if nb is None:
                print(
                    f"skipped {path.name}: its code changed, run it with --execute"
                )
                continue
        changed |= write_or_check(BOOK_SRC / page, render_page(path, nb), check)
    changed |= render_summary(entries, check)
    if failed:
        raise SystemExit(f"notebooks that failed: {', '.join(failed)}")
    return changed


def with_saved_outputs(
    nb: nbformat.NotebookNode, executed: nbformat.NotebookNode
) -> nbformat.NotebookNode | None:
    """Returns ``nb`` with the outputs of ``executed``, or None when their
    code cells differ. The setup cell is ignored, since it installs packages
    only where they are missing and the page leaves it out."""
    code = [c for c in nb.cells if c.cell_type == "code"]
    ran = [c for c in executed.cells if c.cell_type == "code"]

    def sources(cells: list) -> list[str]:
        return [c.source for c in cells if "bootstrap" not in tags(c)]

    if sources(code) != sources(ran):
        return None
    nb = copy.deepcopy(nb)
    for cell, old in zip(
        (c for c in nb.cells if c.cell_type == "code"), ran, strict=True
    ):
        cell.outputs = old.outputs
    return nb


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("bootstrap", "export", "render"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--check", action="store_true")
    render_cmd = sub.choices["render"]
    render_cmd.add_argument("notebooks", nargs="*")
    render_cmd.add_argument("--execute", action="store_true")
    render_cmd.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()

    if args.command == "bootstrap":
        changed = bootstrap(args.check)
    elif args.command == "export":
        changed = export(args.check)
    else:
        changed = render(args.notebooks, args.execute, args.timeout, args.check)
    return 1 if args.check and changed else 0


if __name__ == "__main__":
    sys.exit(main())

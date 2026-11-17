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
"""Draws the book's diagrams as Excalidraw scenes and renders them to SVG.

Each diagram is laid out here with explicit coordinates, so this file is the
source of truth and its diff is reviewable. Every element takes its random
seed from its id, so a diagram renders the same strokes every time. The tool
renders each diagram to ``<name>.svg`` and ``<name>-dark.svg`` in
``book/src/assets/diagrams/`` with the Excalidraw CLI
(``npm install -g @excalidraw/cli``). ``--scenes DIR`` also keeps the
Excalidraw scenes, which open in excalidraw.com for a closer look.

Notebooks show a diagram with a Markdown image whose URL is the published
light SVG, ``https://llm.modular.com/assets/diagrams/<name>.svg``, which
works in Colab and Kaggle. ``nb2book.py`` turns that image into a light and a
dark version that follow the book's theme.

Usage: ``python tools/diagrams.py [--scenes DIR] [name ...]``
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SVGS = ROOT / "book" / "src" / "assets" / "diagrams"

FONT_SIZE = 18
# Excalifont's average advance, as a fraction of the font size. Text is
# placed by its center, so this only sizes boxes; it doesn't shift labels.
CHAR_WIDTH = 0.56
LINE_HEIGHT = 1.25
INK = "#1e1e1e"

# The same stage keeps the same color in every diagram.
STYLES = {
    "io": ("#1971c2", "#e7f5ff"),
    "model": ("#f08c00", "#fff3bf"),
    "sample": ("#2f9e44", "#d3f9d8"),
    "memory": ("#9c36b5", "#f3d9fa"),
    "plain": (INK, "transparent"),
}


def seed(key: str) -> int:
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big")


def text_size(text: str, size: int) -> tuple[float, float]:
    lines = text.split("\n")
    width = max(len(line) for line in lines) * size * CHAR_WIDTH
    return width, len(lines) * size * LINE_HEIGHT


@dataclass
class Diagram:
    name: str
    elements: list[dict] = field(default_factory=list)

    def _base(self, kind: str, key: str, x: float, y: float) -> dict:
        return {
            "id": key,
            "type": kind,
            "x": x,
            "y": y,
            "angle": 0,
            "strokeColor": INK,
            "backgroundColor": "transparent",
            "fillStyle": "solid",
            "strokeWidth": 2,
            "strokeStyle": "solid",
            "roughness": 1,
            "opacity": 100,
            "groupIds": [],
            "frameId": None,
            "index": f"a{len(self.elements):04d}",
            "roundness": None,
            "seed": seed(f"{self.name}/{key}"),
            "version": 1,
            "versionNonce": seed(f"{self.name}/{key}/nonce"),
            "isDeleted": False,
            "boundElements": None,
            "updated": 1,
            "created": 1,
            "link": None,
            "locked": False,
        }

    def text(
        self,
        key: str,
        text: str,
        cx: float,
        cy: float,
        color: str = INK,
        size: int = FONT_SIZE,
    ) -> None:
        """Places ``text`` centered on ``(cx, cy)``."""
        width, height = text_size(text, size)
        element = self._base("text", key, cx - width / 2, cy - height / 2)
        element.update(
            width=width,
            height=height,
            strokeColor=color,
            text=text,
            originalText=text,
            fontSize=size,
            fontFamily=5,
            textAlign="center",
            verticalAlign="middle",
            containerId=None,
            autoResize=True,
            lineHeight=LINE_HEIGHT,
        )
        self.elements.append(element)

    def box(
        self,
        key: str,
        label: str,
        cx: float,
        cy: float,
        style: str = "plain",
        width: float | None = None,
        height: float = 50,
        dashed: bool = False,
        size: int = FONT_SIZE,
    ) -> tuple[float, float, float, float]:
        """Draws a labeled box centered on ``(cx, cy)`` and returns its
        left, top, right and bottom edges."""
        stroke, fill = STYLES[style]
        text_width, _ = text_size(label, size)
        width = width or text_width + 40
        element = self._base("rectangle", key, cx - width / 2, cy - height / 2)
        element.update(
            width=width,
            height=height,
            strokeColor=stroke,
            backgroundColor=fill,
            strokeStyle="dashed" if dashed else "solid",
        )
        self.elements.append(element)
        self.text(f"{key}-label", label, cx, cy, color=stroke, size=size)
        return cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2

    def arrow(
        self,
        key: str,
        points: list[tuple[float, float]],
        both: bool = False,
        dashed: bool = False,
    ) -> None:
        """Draws a polyline through ``points`` with a head at the end, and
        at the start too when ``both`` is set."""
        x0, y0 = points[0]
        rel = [[x - x0, y - y0] for x, y in points]
        xs = [p[0] for p in rel]
        ys = [p[1] for p in rel]
        element = self._base("arrow", key, x0, y0)
        element.update(
            width=max(xs) - min(xs),
            height=max(ys) - min(ys),
            points=rel,
            strokeStyle="dashed" if dashed else "solid",
            startBinding=None,
            endBinding=None,
            startArrowhead="arrow" if both else None,
            endArrowhead="arrow",
            elbowed=False,
        )
        self.elements.append(element)

    def scene(self) -> dict:
        return {
            "type": "excalidraw",
            "version": 2,
            "source": "https://llm.modular.com",
            "elements": self.elements,
            "appState": {"viewBackgroundColor": "#ffffff", "gridSize": 20},
            "files": {},
        }


def row(
    d: Diagram,
    stages: list[tuple[str, str, str]],
    y: float,
    gap: float = 50,
    x: float = 0,
) -> dict[str, tuple[float, float, float, float]]:
    """Draws boxes left to right, joined by arrows, and returns their edges
    by key."""
    edges = {}
    previous = None
    for key, label, style in stages:
        width = text_size(label, FONT_SIZE)[0] + 40
        edges[key] = d.box(key, label, x + width / 2, y, style)
        if previous is not None:
            d.arrow(
                f"{previous}-{key}", [(edges[previous][2] + 6, y), (x - 6, y)]
            )
        previous = key
        x += width + gap
    return edges


def generation_loop() -> Diagram:
    """Section 1.1: one token at a time, appended to the sequence."""
    d = Diagram("generation-loop")
    e = row(
        d,
        [
            ("text", "text", "io"),
            ("tokenizer", "tokenizer", "io"),
            ("ids", "token IDs", "io"),
            ("model", "model", "model"),
            ("logits", "logits", "model"),
            ("sampler", "sampler", "sample"),
            ("next", "next token", "sample"),
        ],
        y=0,
    )
    nx = (e["next"][0] + e["next"][2]) / 2
    ix = (e["ids"][0] + e["ids"][2]) / 2
    low = 80
    d.arrow(
        "loop",
        [(nx, e["next"][3] + 6), (nx, low), (ix, low), (ix, e["ids"][3] + 6)],
    )
    d.text(
        "loop-label",
        "append to the sequence and repeat",
        (nx + ix) / 2,
        low + 20,
    )
    return d


def request_lifecycle() -> Diagram:
    """Section 5.1: the serving loop. The KV cache manager sits beside the
    loop: the scheduler allocates and releases blocks, and the model step
    reads and writes them. New tokens go back to their requests."""
    d = Diagram("request-lifecycle")
    e = row(
        d,
        [
            ("requests", "requests", "io"),
            ("scheduler", "scheduler", "io"),
            ("model", "model step", "model"),
            ("sampler", "sampler", "sample"),
        ],
        y=0,
        gap=110,
    )
    d.text(
        "batch", "batch", (e["scheduler"][2] + e["model"][0]) / 2, -18, size=16
    )
    d.text(
        "logits", "logits", (e["model"][2] + e["sampler"][0]) / 2, -18, size=16
    )

    sx = (e["scheduler"][0] + e["scheduler"][2]) / 2
    mx = (e["model"][0] + e["model"][2]) / 2
    kv = d.box(
        "kv",
        "KV cache manager",
        (sx + mx) / 2,
        -150,
        "memory",
        width=mx - sx + 120,
    )
    d.arrow(
        "sched-kv", [(sx, e["scheduler"][1] - 6), (sx, kv[3] + 6)], both=True
    )
    d.text(
        "sched-kv-label", "allocate and\nrelease blocks", sx - 75, -75, size=16
    )
    d.arrow("model-kv", [(mx, e["model"][1] - 6), (mx, kv[3] + 6)], both=True)
    d.text(
        "model-kv-label",
        "read and write\nkeys and values",
        mx + 80,
        -75,
        size=16,
    )

    qx = (e["sampler"][0] + e["sampler"][2]) / 2
    low = 85
    d.arrow(
        "tokens",
        [
            (qx, e["sampler"][3] + 6),
            (qx, low),
            (sx, low),
            (sx, e["scheduler"][3] + 6),
        ],
    )
    d.text(
        "tokens-label", "a new token for each request", (qx + sx) / 2, low + 20
    )
    return d


def grouped_query_attention() -> Diagram:
    """Section 1.5: SmolLM2's 9 query heads share 3 key/value heads."""
    d = Diagram("grouped-query-attention")
    styles = ["io", "model", "sample"]
    step, group_gap = 70, 40
    centers = []
    x = 0.0
    for head in range(9):
        if head and head % 3 == 0:
            x += group_gap
        centers.append(x)
        d.box(f"q{head}", f"q{head}", x, 0, styles[head // 3], width=56)
        x += step
    for group in range(3):
        gx = sum(centers[group * 3 : group * 3 + 3]) / 3
        d.box(
            f"kv{group}",
            f"k{group}, v{group}",
            gx,
            150,
            styles[group],
            width=120,
        )
        for head in range(group * 3, group * 3 + 3):
            d.arrow(f"q{head}-kv{group}", [(centers[head], 31), (gx, 119)])
    d.text("q-label", "9 query heads", centers[4], -55)
    d.text(
        "kv-label",
        "3 key/value heads, each shared by 3 query heads",
        centers[4],
        205,
    )
    return d


def tensor_parallel_mlp() -> Diagram:
    """Section 2.2: a column-parallel then row-parallel MLP on 3 devices."""
    d = Diagram("tensor-parallel-mlp")
    rows = [-110, 0, 110]
    for i, y in enumerate(rows):
        d.text(f"dev{i}", f"device {i}", -40, y, size=16)
        d.box(f"x{i}", "x", 60, y, "io", width=60)
        d.box(
            f"gu{i}",
            f"gate, up\ncolumns {i}",
            220,
            y,
            "model",
            width=150,
            height=60,
        )
        d.box(
            f"down{i}", f"down\nrows {i}", 410, y, "model", width=130, height=60
        )
        d.box(f"part{i}", f"partial y{i}", 590, y, "memory", width=140)
        d.arrow(f"x{i}-gu{i}", [(96, y), (139, y)])
        d.arrow(f"gu{i}-down{i}", [(301, y), (339, y)])
        d.arrow(f"down{i}-part{i}", [(481, y), (514, y)])
    ar = d.box(
        "allreduce",
        "all-reduce\n(sum)",
        790,
        0,
        "sample",
        width=130,
        height=280,
    )
    for i, y in enumerate(rows):
        d.arrow(f"part{i}-ar", [(666, y), (ar[0] - 6, y)])
    d.box("y", "y", 930, 0, "io", width=60)
    d.arrow("ar-y", [(ar[2] + 6, 0), (894, 0)])
    d.text("note-in", "every device holds all of x", 60, 190, size=16)
    d.text(
        "note-mid",
        "each device computes a slice of the\nintermediate, then a partial sum",
        410,
        195,
        size=16,
    )
    return d


def mixture_of_experts() -> Diagram:
    """Section 6.3: a router picks the top experts for each token."""
    d = Diagram("mixture-of-experts")
    d.box("h", "hidden state", 0, 0, "io")
    d.box("router", "router", 190, 0, "model")
    d.arrow("h-router", [(76, 0), (143, 0)])
    experts = [
        ("e1", "expert 1", True),
        ("e2", "expert 2", False),
        ("e3", "expert 3", True),
        ("dots", "...", False),
        ("e127", "expert 127", True),
        ("e128", "expert 128", False),
    ]
    ys = [-200 + 80 * k for k in range(len(experts))]
    for (key, label, chosen), y in zip(experts, ys, strict=True):
        style = "sample" if chosen else "plain"
        d.box(key, label, 450, y, style, width=150, dashed=not chosen)
        if chosen:
            d.arrow(f"router-{key}", [(237, 0), (369, y)])
            d.arrow(f"{key}-sum", [(531, y), (659, 0)])
    d.text("pick", "top 8 of 128\nexperts run", 175, 70, size=16)
    d.box("sum", "weighted sum", 740, 0, "sample")
    d.text(
        "weights",
        "weights: the chosen\nrouter scores, renormalized",
        740,
        75,
        size=16,
    )
    d.box("out", "output", 920, 0, "io")
    d.arrow("sum-out", [(813, 0), (873, 0)])
    return d


def speculative_decoding() -> Diagram:
    """Section 5.5: draft k tokens, verify them in one target step."""
    d = Diagram("speculative-decoding")
    d.text("draft-label", "draft model\nk steps", -90, 0, size=16)
    tokens = ["t1", "t2", "t3", "t4"]
    xs = [60 + 110 * k for k in range(4)]
    for k, (tok, x) in enumerate(zip(tokens, xs, strict=True)):
        d.box(f"d{tok}", tok, x, 0, "io", width=70)
        if k:
            d.arrow(f"d{k}", [(xs[k - 1] + 41, 0), (x - 41, 0)])
    d.text("target-label", "target model\n1 step", -90, 140, size=16)
    d.box(
        "verify",
        "check t1 to t4, and predict one more",
        (xs[0] + xs[-1]) / 2 + 55,
        140,
        "model",
        width=520,
    )
    d.arrow(
        "to-verify", [(xs[-1] + 41, 0), (xs[-1] + 80, 0), (xs[-1] + 80, 105)]
    )
    d.text("result-label", "kept", -90, 280, size=16)
    outcome = [("t1", "sample"), ("t2", "sample"), ("t3'", "model")]
    for k, (tok, style) in enumerate(outcome):
        d.box(f"r{k}", tok, xs[k], 280, style, width=70)
    d.arrow(
        "verify-result",
        [((xs[0] + xs[2]) / 2, 171), ((xs[0] + xs[2]) / 2, 249)],
    )
    d.text(
        "result-note",
        "t1 and t2 match the target and are kept. At t3 the target\n"
        "disagrees: its own token t3' takes t3's place, and t4 is dropped.",
        (xs[0] + xs[-1]) / 2 + 55,
        360,
        size=16,
    )
    return d


DIAGRAMS = {
    "grouped-query-attention": grouped_query_attention,
    "tensor-parallel-mlp": tensor_parallel_mlp,
    "mixture-of-experts": mixture_of_experts,
    "speculative-decoding": speculative_decoding,
    "generation-loop": generation_loop,
    "request-lifecycle": request_lifecycle,
}


def render(name: str, scenes: Path) -> None:
    SVGS.mkdir(parents=True, exist_ok=True)
    scene = scenes / f"{name}.excalidraw"
    scene.write_text(json.dumps(DIAGRAMS[name]().scene(), indent=1) + "\n")
    for suffix, theme in (("", "light"), ("-dark", "dark")):
        subprocess.run(
            [
                "excalidraw",
                "render",
                str(scene),
                "--out",
                str(SVGS / f"{name}{suffix}.svg"),
                "--theme",
                theme,
                "--transparent",
            ],
            check=True,
            capture_output=True,
        )
        print(f"wrote {SVGS.relative_to(ROOT) / f'{name}{suffix}.svg'}")


def main() -> int:
    args = sys.argv[1:]
    keep = None
    if args[:1] == ["--scenes"]:
        keep, args = Path(args[1]), args[2:]
        keep.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for name in args or list(DIAGRAMS):
            render(name, keep or Path(tmp))
    return 0


if __name__ == "__main__":
    sys.exit(main())

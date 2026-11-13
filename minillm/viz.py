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
"""Gives the book's interactive figures one look.

Importing this module registers a Plotly template named ``minillm`` and makes
it the default. The colors are the light-mode steps of a palette checked for
color-vision deficiency. The book's page script swaps each one for its
dark-mode step, so every hex here must stay lowercase and listed in
``book/theme/plotly-figures.js``.
"""

from __future__ import annotations

import html
from collections.abc import Sequence

import plotly.graph_objects as go
import plotly.io as pio
from IPython.display import Markdown
from plotly.basedatatypes import BaseTraceType

# Categorical colors in a fixed order; adjacent pairs stay distinct under
# color-vision deficiency. Use them in this order, never cycled.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
)
SERIES = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e6e5e1"
MUTED = "#b5b3ad"  # inactive marks, such as filtered-out tokens
SURFACE = "#fcfcfb"
NEUTRAL = "#f0efec"

# One hue, light to dark, for magnitudes such as attention weights.
SEQUENTIAL = [
    [0.0, "#f0efec"],
    [0.15, "#cde2fb"],
    [0.35, "#86b6ef"],
    [0.55, "#3987e5"],
    [0.75, "#256abf"],
    [1.0, "#104281"],
]
# Two hues around a gray midpoint, for signed values such as weights.
DIVERGING = [
    [0.0, "#104281"],
    [0.25, "#5598e7"],
    [0.5, "#f0efec"],
    [0.75, "#e66767"],
    [1.0, "#a52a2a"],
]

# Passed to `fig.show(config=CONFIG)` and used by the book's page script.
CONFIG = {"displaylogo": False, "responsive": True}

_axis = {
    "gridcolor": GRID,
    "automargin": True,
    "linecolor": GRID,
    "zeroline": False,
    "ticks": "",
    "title": {"font": {"color": INK_SECONDARY}},
    "tickfont": {"color": INK_SECONDARY},
}

pio.templates["minillm"] = go.layout.Template(
    layout={
        "colorway": SERIES,
        "font": {
            "family": "Inter, system-ui, sans-serif",
            "size": 13,
            "color": INK,
        },
        "title": {"font": {"size": 16}, "x": 0, "xref": "paper"},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "xaxis": _axis,
        "yaxis": _axis,
        "colorscale": {"sequential": SEQUENTIAL, "diverging": DIVERGING},
        "hoverlabel": {
            "bgcolor": SURFACE,
            "bordercolor": GRID,
            "font": {"color": INK},
        },
        "legend": {"orientation": "h", "y": 1.02, "yanchor": "bottom", "x": 0},
        "margin": {"l": 56, "r": 24, "t": 64, "b": 48},
        "sliders": [{"currentvalue": {"font": {"color": INK_SECONDARY}}}],
    },
    data={
        "scatter": [go.Scatter(line={"width": 2}, marker={"size": 8})],
        "bar": [go.Bar(marker={"line": {"width": 0}})],
    },
)
pio.templates.default = "minillm"


def show(fig: go.Figure) -> None:
    """Displays a figure with the book's settings.

    Args:
        fig: The figure to display.
    """
    fig.show(config=CONFIG)


def with_slider(
    fig: go.Figure,
    steps: Sequence[Sequence[BaseTraceType]],
    labels: Sequence[str],
    prefix: str,
    active: int = 0,
) -> go.Figure:
    """Adds a slider that shows one group of traces at a time.

    Traces already in ``fig`` stay visible at every step. Each step only
    toggles visibility, so the slider also works in the book's static pages.

    Args:
        fig: The figure to add the traces and the slider to.
        steps: One group of traces per slider step.
        labels: The label of each step.
        prefix: The text before the current label, such as ``"head "``.
        active: The step shown first.

    Returns:
        The figure, for chaining.
    """
    first = len(fig.data)
    for index, group in enumerate(steps):
        for trace in group:
            trace.visible = index == active
            fig.add_trace(trace)
    indices = list(range(first, len(fig.data)))
    slider_steps = []
    start = 0
    for label, group in zip(labels, steps, strict=True):
        visible = [False] * len(indices)
        visible[start : start + len(group)] = [True] * len(group)
        start += len(group)
        slider_steps.append(
            {
                "label": label,
                "method": "restyle",
                "args": [{"visible": visible}, indices],
            }
        )
    fig.update_layout(
        sliders=[
            {
                "active": active,
                "steps": slider_steps,
                "currentvalue": {"prefix": prefix},
                "pad": {"t": 56},
                "x": 0.03,
                "len": 0.94,
            }
        ]
    )
    return fig


# Translucent chip colors that read on light and dark pages alike.
CHIP_COLORS = [
    "rgba(42,120,214,0.22)",
    "rgba(235,104,52,0.22)",
    "rgba(27,175,122,0.22)",
    "rgba(237,161,0,0.25)",
    "rgba(232,123,164,0.25)",
    "rgba(98,80,214,0.22)",
]


def tokens(
    pieces: Sequence[str],
    ids: Sequence[int],
    colors: Sequence[str] | None = None,
    titles: Sequence[str] | None = None,
) -> Markdown:
    """Shows text split into tokens, one colored chip per token.

    Hovering a chip shows its title. Spaces show as a middle dot, since
    most tokens start with one.

    Args:
        pieces: The text of each token.
        ids: The ID of each token.
        colors: A CSS color per chip, such as one of :data:`CHIP_COLORS`.
            By default the chips cycle through them to show the boundaries.
        titles: Hover text per chip. By default, the token ID.

    Returns:
        A Markdown display object for the notebook and the book.
    """
    chips = []
    for index, (piece, token_id) in enumerate(zip(pieces, ids, strict=True)):
        color = (
            colors[index] if colors else CHIP_COLORS[index % len(CHIP_COLORS)]
        )
        title = html.escape(titles[index]) if titles else f"token ID {token_id}"
        text = (
            html.escape(piece).replace(" ", "&middot;").replace("\n", "&crarr;")
        )
        chips.append(
            f'<span title="{title}" style="background:{color};'
            "border-radius:4px;padding:2px 4px;margin:0 2px 4px 0;display:inline-block;"
            f'font-family:ui-monospace,monospace;font-size:0.9em">{text}</span>'
        )
    return Markdown(f'<div style="line-height:2">{"".join(chips)}</div>')

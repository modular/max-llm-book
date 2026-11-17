// Renders the interactive figures that `tools/nb2book.py` saves from the
// notebooks. Each figure is a `<div class="plotly-figure" data-src="...">`
// pointing at Plotly JSON. Plotly loads only on pages that have a figure.
//
// The figures carry the light-mode colors of `minillm/viz.py`. In a dark
// theme, every one of them is swapped for its dark-mode step in one pass.
(() => {
  const PLOTLY = "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@4.1.1/plotly.min.js";
  const DARK = {
    // Categorical.
    "#2a78d6": "#3987e5",
    "#eb6834": "#d95926",
    "#1baf7a": "#199e70",
    "#eda100": "#c98500",
    "#e87ba4": "#d55181",
    "#6250d6": "#9085e9",
    "#e34948": "#e66767",
    // Ink, grid, and surfaces.
    "#0b0b0b": "#f0efec",
    "#52514e": "#c3c2b7",
    "#e6e5e1": "#383835",
    "#b5b3ad": "#5b5a56",
    "#fcfcfb": "#1a1a19",
    // Sequential and diverging ramps: near zero recedes into the surface.
    "#f0efec": "#2b2b29",
    "#cde2fb": "#104281",
    "#86b6ef": "#1c5cab",
    "#3987e5": "#2a78d6",
    "#256abf": "#5598e7",
    "#104281": "#9ec5f4",
    "#5598e7": "#3987e5",
    "#e66767": "#c94a4a",
    "#a52a2a": "#f08a8a",
  };
  const figures = [...document.querySelectorAll(".plotly-figure")];
  if (figures.length === 0) return;

  const isDark = () =>
    ["coal", "navy", "ayu"].some((name) => document.documentElement.classList.contains(name));
  const themed = (json) =>
    isDark() ? JSON.parse(JSON.stringify(json).replace(/#[0-9a-f]{6}/g, (hex) => DARK[hex] ?? hex)) : json;
  const config = { displaylogo: false, responsive: true };

  const draw = (div) => {
    const figure = themed(div.figure);
    window.Plotly.react(div, { ...figure, config });
  };

  const script = document.createElement("script");
  script.src = PLOTLY;
  script.onload = () => {
    for (const div of figures) {
      fetch(div.dataset.src)
        .then((response) => response.json())
        .then((json) => {
          div.figure = json;
          draw(div);
        })
        .catch(() => {
          div.textContent = "This figure could not be loaded.";
        });
    }
    // mdBook switches themes by changing the class on <html>.
    new MutationObserver(() => {
      for (const div of figures) if (div.figure) draw(div);
    }).observe(document.documentElement, { attributes: true, attributeFilter: ["class"] });
  };
  document.head.appendChild(script);
})();

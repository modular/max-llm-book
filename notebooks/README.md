# Chapter notebooks

Each chapter of the [book](https://llm.modular.com) is one notebook here. The
notebooks are the source of truth: the web pages and the `minillm` package are
generated from them.

## Run a chapter

You don't need a GPU: every chapter runs on a CPU. When MAX finds a GPU
(NVIDIA, AMD, or Apple silicon), the notebooks use it automatically.

- **Colab or Kaggle:** use the badges at the top of each chapter page. The
  first cell installs MAX and this repo's package. On Kaggle, turn on
  **Internet** in the notebook settings first.
- **Locally with pixi:** from the repo root, run `pixi run notebook`. The
  first cell detects the pixi environment and installs nothing.
- **Locally with pip:** in a fresh virtual environment, open the notebook in
  Jupyter. The first cell installs the MAX nightly wheel and this repo's
  package from GitHub. To install your local checkout instead, set
  `MAX_LLM_BOOK_SRC=/path/to/max-llm-book` before starting Jupyter.

The first time a chapter loads a model, it downloads the weights from Hugging
Face and compiles the model; later runs reuse both caches.

## Edit a chapter

Edit the notebook, clear its outputs, and then regenerate the derived files:

```bash
pixi run bootstrap  # copy _bootstrap.py into each notebook's first cell
pixi run export     # write `# export:` cells into the minillm package
pixi run render     # run every notebook and write its page to book/src
pixi run test       # includes checks that the generated files are current
```

Commit the notebook and the files these tasks change. `pixi run
test-notebooks` runs every notebook top to bottom, as the nightly CI does.

Cell tags change how a cell appears on the web page:

| Tag            | Effect on the page                           |
|----------------|----------------------------------------------|
| `bootstrap`    | Omitted. Set by `pixi run bootstrap`.        |
| `hide-in-book` | Omitted.                                     |
| `hide-input`   | Shows the output only.                       |
| `hide-output`  | Shows the code only.                         |

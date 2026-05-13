# Interactive notebook

`tutorial.ipynb` is a companion to the [max-llm-book](https://llm.modular.com)
tutorial. It lets you run each GPT-2 component interactively, inspect real
tensor shapes and activations, and generate text from pretrained HuggingFace
weights — all alongside the narrative chapters.

To browse the cell outputs without setting up an environment, open
[`tutorial.rendered.ipynb`](tutorial.rendered.ipynb) — a frozen snapshot with
all plots and printed values preserved. See
[Rendered snapshot](#rendered-snapshot) below for how to regenerate it.

## Prerequisites

- [pixi](https://pixi.sh/dev/) — install with
  `curl -fsSL https://pixi.sh/install.sh | sh`
- A platform supported by MAX. See the
  [MAX system requirements](https://docs.modular.com/max/packages#system-requirements).
- About 1 GB of free disk for the HuggingFace cache and pixi env (first run
  only).

A GPU is **not** required. The notebook auto-detects one and uses it if
available; otherwise it runs on CPU (slower, but everything completes).

## Launch

From the repo root:

```bash
pixi install     # one-time, installs dependencies
pixi run notebook
```

This opens JupyterLab with `tutorial.ipynb` in your browser. Run cells
top-to-bottom with **Run → Run All Cells**, or step through them one at a
time.

If you prefer your own Jupyter, register the pixi env as a kernel once and
then select it in the Jupyter UI:

```bash
pixi run python -m ipykernel install --user --name max-llm-book \
  --display-name "Python (max-llm-book)"
```

## Device selection

The setup cell picks a device at runtime:

- **Linux / Windows with a GPU** → `Accelerator()` (default)
- **macOS (Apple Silicon)** → `CPU()`. The Metal backend produces incorrect
  logits for sequence lengths > 3, so the notebook forces CPU there.
- **No accelerator present** → `CPU()`

The setup cell prints the chosen device (`Device : Device(type=gpu,id=0)` or
similar) so you can confirm. No manual flag is needed.

## First-run expectations

| Section | What happens                                                                                                     | Time (GPU / CPU)                |
|---------|------------------------------------------------------------------------------------------------------------------|---------------------------------|
| 1 – 8   | Eager execution with random weights; tensor shape demos and plots                                                | instant                         |
| 9       | Downloads ~500 MB of GPT-2 weights to `~/.cache/huggingface/`, adapts them to MAX layout, and compiles the graph | ~30–60 s / 1–2 min              |
| 10 – 11 | Reads pipeline/KV-cache config — no heavy compute                                                                | instant                         |
| 12      | Autoregressive greedy generation of 20 tokens                                                                    | a few seconds / several minutes |

Section 12 is slow on CPU because this tutorial intentionally reprocesses the
full token sequence on every decode step (no incremental KV cache — see
Step 10).

## Troubleshooting

- **Kernel not found when opening from a global Jupyter** — either launch with
  `pixi run notebook`, or register the env once with the `ipykernel install`
  command above and pick *Python (max-llm-book)* in the kernel menu.
- **HuggingFace download fails or is slow** — retry, optionally with
  `HF_HUB_ENABLE_HF_TRANSFER=1 pixi run notebook` for faster parallel
  downloads. The cache at `~/.cache/huggingface/` is reused across runs.
- **Need to reset notebook state** — *Kernel → Restart Kernel and Run All*.
  This re-creates the device context and clears any leaked tensor state.
- **Out of GPU memory on Section 9** — close other GPU workloads
  (`nvidia-smi` to check), then restart the kernel. On a machine with <4 GB
  free VRAM, temporarily force CPU by editing the setup cell
  (`DEVICE = CPU()`).

## Headless validation

To execute every cell from the command line (useful for CI or a quick
end-to-end check):

```bash
pixi run jupyter nbconvert \
  --to notebook \
  --execute notebooks/tutorial.ipynb \
  --output /tmp/tutorial-executed.ipynb \
  --ExecutePreprocessor.timeout=1800 \
  --ExecutePreprocessor.kernel_name=python3
```

The command exits 0 on success. Open `/tmp/tutorial-executed.ipynb` in
JupyterLab to review the outputs — the Step 12 cell should show an English
continuation of "In the beginning".

## Rendered snapshot

`tutorial.rendered.ipynb` is a checked-in copy of the notebook with all
outputs preserved so readers can browse plots, tensor shapes, and generated
text directly on GitHub. It is a static artifact — edit `tutorial.ipynb`
instead, then regenerate the snapshot:

```bash
pixi run jupyter nbconvert \
  --to notebook \
  --execute notebooks/tutorial.ipynb \
  --output tutorial.rendered.ipynb \
  --ExecutePreprocessor.timeout=1800 \
  --ExecutePreprocessor.kernel_name=python3
```

Then re-add the "Rendered snapshot" banner as the first cell (date + MAX
nightly version) so readers know what build produced the outputs.

## Read the book

```bash
pixi run book
```

Or online at [llm.modular.com](https://llm.modular.com).

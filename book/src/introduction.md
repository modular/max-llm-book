# Build and serve an LLM with MAX

This book takes you from zero to serving your own language model with
[MAX](https://max.modular.com/), Modular's framework for AI inference. You
start by running a real model in three lines of Python. Then you rebuild
that model from scratch, check every part against MAX's own implementation,
and go on to the techniques that production inference engines use:
distributing a model across devices, writing your own Mojo kernels, testing
and benchmarking, building a serving engine, bringing up newer models, and
serving your own architecture.

## Why this book

Most explanations of LLM inference stop at diagrams, and most inference
engines are too large to read. This book sits between the two. You write
every piece yourself, small enough to read in one sitting, and you run it on
a real model so that each claim is a number you measured: the logits match
the reference, the KV cache stops each new token from recomputing the whole
text, batching raises throughput.

You do it with MAX's own APIs, the same ones its production models use, so
what you learn carries over directly:

- `max.pipelines`: `AutoPipeline`, `AutoModel`, and `AutoTokenizer` for
  running models from Python.
- `max.tensor` and `max.functional`: tensors and operations that run eagerly,
  like NumPy, and compile into fast graphs.
- `max.nn`: modules and layers for building models.
- Mojo custom ops: kernels you write yourself and call from Python.
- `max.graph`: the graph API that `compile` builds on, for full control.
- `max serve`, `max generate`, and `max benchmark`: the command-line tools
  for serving, testing, and measuring models.

Every section is a notebook with interactive figures you can hover, zoom,
and slide through, and every notebook checks its own results.

## The model

The book uses
[SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct).
It is a real chat model, small enough to run on a laptop CPU, and it uses the
Llama architecture that most open models share: RMSNorm, rotary position
embeddings, grouped-query attention, and a SwiGLU MLP. What you build for it
applies to Llama, Qwen, Mistral, and their relatives, and [Chapter 6](./ch06.md)
extends it to Qwen3, Gemma 3, and mixture-of-experts models.

## Get started

<div class="callout">
<p class="callout-title">Check your hardware first</p>
<p>Before you install anything, check that MAX supports your machine and GPU
in the
<a href="https://max.modular.com/system-requirements/">MAX
system requirements</a>. MAX runs on Linux and on macOS with Apple silicon.
On NVIDIA, the notebooks need an Ampere-generation GPU or newer; on an older
one, such as a T4 or P100, they run on the CPU instead.</p>
</div>

You can run the notebooks in Colab or Kaggle with nothing installed, or on
your own machine.

### In Colab or Kaggle

Every section page starts with Open in Colab and Open in Kaggle buttons. The
first cell of every notebook installs what the notebook needs, so run the cells
in order from the top:

```python
# The setup cell, shortened: it installs MAX and this book's package
# with uv, but only if they're missing.
if importlib.util.find_spec("max") is None:
    uv_install("max[all]")  # from https://whl.modular.com/nightly/simple/
if importlib.util.find_spec("minillm") is None:
    uv_install("git+https://github.com/modular/max-llm-book@main")
```

`max[all]` is the MAX package with everything the book uses: the Python
APIs, the `max` and `mojo` commands, serving, and the Jupyter `%%mojo`
magic. `minillm` is the package you build in the book, installed so that
later sections can import what earlier ones wrote.

Choose the runtime before you run the first cell:

- In Colab, the CPU runtime works for every notebook. For a GPU, choose an
  L4 or A100 runtime. MAX's matrix multiplication needs an NVIDIA GPU from
  the Ampere generation or later, so on a T4 runtime the setup cell hides
  the GPU and the notebook runs on the CPU.
- In Kaggle, turn on Internet in the notebook settings so that the setup
  cell can download packages. Kaggle's GPUs, the T4 and the P100, are older
  than Ampere, so the notebooks run on the CPU there.

### On your own machine

Install with pixi or uv, on a machine that meets the
[system requirements](https://max.modular.com/system-requirements/).

With [pixi](https://pixi.sh), which installs Python, MAX, and every other
dependency into a project environment:

1. Install pixi:

   ```bash
   curl -fsSL https://pixi.sh/install.sh | sh
   ```

2. Clone the book:

   ```bash
   git clone https://github.com/modular/max-llm-book
   cd max-llm-book
   ```

3. Open the notebooks in JupyterLab:

   ```bash
   pixi run notebook
   ```

With [uv](https://docs.astral.sh/uv/) and Python 3.10 or later:

1. Create and activate an environment:

   ```bash
   uv venv --python 3.12
   source .venv/bin/activate
   ```

2. Install MAX from the nightly package index:

   ```bash
   uv pip install --prerelease allow --index-strategy unsafe-best-match \
     --extra-index-url https://whl.modular.com/nightly/simple/ "max[all]"
   ```

3. Clone the book and install its package, with JupyterLab:

   ```bash
   git clone https://github.com/modular/max-llm-book
   cd max-llm-book
   uv pip install -e . jupyterlab
   ```

4. Open the notebooks:

   ```bash
   jupyter lab notebooks
   ```

Check the installation, and whether MAX found a GPU:

```bash
max --version
python -c "from max.driver import accelerator_count; print('GPUs:', accelerator_count())"
```

The book uses the MAX nightly build, because some of the APIs it teaches are
newer than the last stable release. Every notebook runs on a CPU and uses a GPU
automatically when MAX finds a supported one.

<div class="callout">
<p class="callout-title">About the numbers</p>
<p>Timings, speedups, and other measurements on these pages come from one
run on an Apple-silicon Mac with a Metal GPU. Treat them as examples. Your
numbers depend on your hardware and vary from run to run; the notebooks
check results, such as matching logits, that hold on every machine.</p>
</div>

## Chapters

| Chapter                                  | What you do                                                                                                                                                                        |
|------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [1. Foundations](./ch01.md)              | Run and serve SmolLM2, then rebuild it from scratch: tensors, weights, layers, attention, the full model checked against MAX's, sampling, and a KV cache.                          |
| [2. Distributed tensors](./ch02.md)      | Shard tensors and the model across a device mesh, simulated on one machine, and check that the result doesn't change.                                                              |
| [3. Custom ops in Mojo](./ch03.md)       | Write a kernel in Mojo, call it from your model, and see the graph API that `compile` builds on.                                                                                   |
| [4. Testing and benchmarking](./ch04.md) | Test layers against references, verify logits, and benchmark and profile models and kernels.                                                                                       |
| [5. A serving engine](./ch05.md)         | Build the core of an inference engine: continuous batching, a paged KV cache, prefix caching, and speculative decoding.                                                            |
| [6. Bigger models](./ch06.md)            | Extend your model to Qwen3, Gemma 3, and mixture-of-experts.                                                                                                                       |
| [7. Production serving](./ch07.md)       | Serve your own architecture with `max serve`, benchmark it, use structured output and tools, and call state-of-the-art open models on [Modular's cloud](https://www.modular.com/). |

Read the chapters in order the first time: each section builds on code from
earlier ones, which the `minillm` package collects for you.

## The handbook

The [Modular handbook](https://handbook.modular.com/) explains how LLM
inference works in production: metrics, batching, KV cache management,
quantization, and GPU kernels. This book is its hands-on companion. Each
section links the handbook pages for the ideas it implements, so you can
read the concept and then build it.

## Prerequisites

You should be comfortable with Python and NumPy-style array code. Knowing
what a neural network layer is helps; the book explains everything specific
to transformers as it goes. [Section 6.2](./06_02_gemma3.md) uses Gemma 3,
whose weights require you to accept Google's license on Hugging Face and log
in with `hf auth login`.

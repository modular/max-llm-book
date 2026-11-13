# Build and serve an LLM with MAX

A hands-on book that takes you from running a model in three lines to
building and serving your own. You rebuild
[SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)
from tensors up with MAX's eager APIs, check every step against reference
implementations, write a serving engine, bring up newer models, and call
state-of-the-art open models on Modular's cloud.
Read it at [llm.modular.com](https://llm.modular.com/).

Every section is a notebook that runs on your laptop, in Google Colab, or
on Kaggle. MAX uses a GPU when it finds a supported one, and the CPU
otherwise.

## Chapters

1. [Foundations](https://llm.modular.com/ch01.html): run and serve SmolLM2,
   then rebuild it: tensors and eager mode, weights, layers, attention, the
   full model, sampling, and a KV cache.
2. [Distributed tensors](https://llm.modular.com/ch02.html): shard tensors
   and the model across a device mesh, simulated on one machine.
3. [Custom ops in Mojo](https://llm.modular.com/ch03.html): write a kernel in
   a `%%mojo` cell, call it with `F.custom`, and build graphs with the graph
   API.
4. [Testing and benchmarking](https://llm.modular.com/ch04.html): layer
   tests, logit verification, kernel benchmarks, and profiling.
5. [A serving engine](https://llm.modular.com/ch05.html): continuous
   batching, a paged KV cache, prefix caching, and speculative decoding.
6. [Bigger models](https://llm.modular.com/ch06.html): Qwen3, Gemma 3, and
   mixture of experts.
7. [Production serving](https://llm.modular.com/ch07.html): serve your own
   architecture with `max serve`, benchmark it, use structured output and
   tools, and call state-of-the-art open models on
   [Modular's cloud](https://www.modular.com/).

The book links the [Modular handbook](https://handbook.modular.com/) for the
ideas behind each section.

## Run the notebooks

In Colab or Kaggle, open a notebook from its badge on
[llm.modular.com](https://llm.modular.com/). Its first cell installs MAX and
this repository's `minillm` package with `uv`.

Locally, install [Pixi](https://pixi.sh/) and check the
[MAX system requirements](https://max.modular.com/system-requirements/):

```bash
git clone https://github.com/modular/max-llm-book
cd max-llm-book
pixi run notebook
```

## Repository layout

```text
notebooks/   one notebook per section, the source of truth
minillm/     the package the notebooks build, generated from their export cells
book/        the mdBook site; pages in book/src are generated from the notebooks
tools/       nb2book.py, which generates minillm/ and the pages
tests/       unit tests, sync checks, and papermill tests of every notebook
```

## Develop

Edit a notebook, then regenerate everything derived from it:

```bash
pixi run render                # export minillm/, run the notebooks, write the pages
pixi run test                  # unit tests and sync checks
pixi run test-notebooks        # run every notebook with papermill
pixi run test-notebooks-smoke  # the few notebooks CI runs
pixi run book                  # preview the site
```

## License

Apache License v2.0 with LLVM Exceptions. See [LICENSE](LICENSE).

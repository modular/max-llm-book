# Build an LLM from scratch with MAX

A guided tour of a complete GPT-2 implementation using the MAX framework.
Each section walks through the code in `gpt2_arch/gpt2.py` and explains what
it does and why — from model configuration through serving with `max serve`.

## What you'll learn

- **Transformer architecture**: Every component of GPT-2, explained through
  working code
- **MAX Python API**: How MAX's `experimental.nn` builds and compiles neural
  networks
- **Inference patterns**: Weight loading, lazy initialization, model
  compilation, and autoregressive generation

## Quick start

### Prerequisites

- [Pixi](https://pixi.sh/) package manager
- Basic understanding of neural networks
- You'll need to meet the
  [MAX system requirements](https://docs.modular.com/max/packages#system-requirements)

### Installation

```bash
git clone https://github.com/modular/max-llm-book
cd max-llm-book
pixi install
```

### Run the model

Serve GPT-2 via an OpenAI-compatible HTTP endpoint:

```bash
pixi run serve
```

Then query it:

```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"openai-community/gpt2","prompt":"In the beginning","max_tokens":30,"temperature":0}'
```

### Run the notebook

Explore each GPT-2 component interactively — real tensor shapes, activation
visualizations, and live text generation from pretrained weights:

```bash
pixi run notebook
```

This opens JupyterLab with `notebooks/tutorial.ipynb`. Sections 1–8 run
immediately with random weights; sections 9–12 download the pretrained GPT-2
checkpoint (~500 MB) from Hugging Face and compile the model for inference.

### Read the book

```bash
pixi run book
```

Or read it online at [llm.modular.com](https://llm.modular.com/).

## What the book covers

The tutorial walks through `gpt2_arch/` section by section:

| Section | Topic                       | What you'll learn                                                    |
|---------|-----------------------------|----------------------------------------------------------------------|
| —       | Run the model               | Serve GPT-2 with `pixi run serve` before diving into code            |
| 1       | Model configuration         | Architecture hyperparameters and Hugging Face compatibility          |
| 2       | Feed-forward network        | Two-layer MLP with GELU activation                                   |
| 3       | Causal masking              | Preventing attention to future tokens                                |
| 4       | Multi-head attention        | Parallel attention across 12 heads                                   |
| 5       | Layer normalization         | Pre-norm pattern for stable activations                              |
| 6       | Transformer block           | Residual connections and component wiring                            |
| 7       | Stack transformer blocks    | Embeddings and the 12-layer model body                               |
| 8       | Language model head         | Projecting hidden states to vocabulary logits                        |
| 9       | Weight adaptation           | Reconciling the HuggingFace checkpoint with MAX's weight layout      |
| 10      | KV cache configuration      | Exposing attention dimensions for cache pre-allocation               |
| 11      | Pipeline model              | Load, compile, and execute the model inside `max serve`              |
| 12      | Architecture registration   | Declare the package to `max serve` and wire all pieces together      |

## Project structure

```text
max-llm-book/
├── book/                  # mdBook tutorial documentation
│   └── src/
│       ├── introduction.md
│       ├── serve_first.md
│       ├── step_01.md ... step_12.md
│       └── SUMMARY.md
├── gpt2_arch/            # GPT-2 model + custom architecture package for `max serve`
│   ├── gpt2.py           # Model definition (GPT2Config through MaxGPT2LMHeadModel)
│   ├── model.py          # PipelineModel wrapper used by max serve
│   ├── weight_adapters.py# HuggingFace → MAX weight conversion
│   ├── model_config.py   # KV cache dimension configuration
│   └── arch.py           # Architecture registration entry point
├── notebooks/            # Interactive Jupyter notebook companion
│   └── tutorial.ipynb
├── tests/                # Tests for gpt2_arch/
├── pixi.toml             # Project dependencies and tasks
└── README.md             # This file
```

## Learning resources

- **MAX documentation**: [docs.modular.com](https://docs.modular.com/)
- **Hugging Face GPT-2**: [huggingface.co/gpt2](https://huggingface.co/gpt2)
- **Attention Is All You Need**:
  [arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
- **Language Models are Unsupervised Multitask Learners** (GPT-2 paper):
  [openai.com](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

## Contributing

Found an issue or want to improve the tutorial? Contributions welcome:

1. File issues for bugs or unclear explanations
2. Suggest improvements to code examples or visualizations
3. Open a pull request with fixes or additions

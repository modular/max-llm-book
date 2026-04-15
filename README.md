# Build an LLM from scratch with MAX

A guided tour of a complete GPT-2 implementation using the MAX framework.
Each section walks through the code in `gpt2.py` and explains what it does and
why—from model configuration through streaming text generation.

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
  -d '{"model":"gpt2","prompt":"In the beginning","max_tokens":30,"temperature":0}'
```

Or run the model directly in your terminal:

```bash
pixi run gpt2
```

This downloads the pretrained GPT-2 weights from Hugging Face, compiles the
model, and starts an interactive prompt where you can enter text and see
generated completions.

Additional modes:

```bash
pixi run gpt2 -- --prompt "In the beginning"   # single generation, then exit
pixi run chat                                    # streaming multi-turn chat
pixi run gpt2 -- --benchmark                    # tokens/sec benchmark
```

### Read the book

```bash
pixi run book
```

Or read it online at [llm.modular.com](https://llm.modular.com/).

## What the book covers

The tutorial walks through `gpt2.py` section by section:

| Section | Topic                         | What you'll learn                                                 |
|---------|-------------------------------|-------------------------------------------------------------------|
| —       | Run the model                 | Serve GPT-2 with `pixi run serve` before diving into code         |
| 1       | Model configuration           | Architecture hyperparameters and Hugging Face compatibility       |
| 2       | Feed-forward network          | Two-layer MLP with GELU activation                                |
| 3       | Causal masking                | Preventing attention to future tokens                             |
| 4       | Multi-head attention          | Parallel attention across 12 heads                                |
| 5       | Layer normalization           | Pre-norm pattern for stable activations                           |
| 6       | Transformer block             | Residual connections and component wiring                         |
| 7       | Stacking transformer blocks   | Embeddings and the 12-layer model body                            |
| 8       | Language model head           | Projecting hidden states to vocabulary logits                     |
| 9       | Tokens, weights, and sampling | BPE tokenization, weight loading, and Gumbel-max sampling         |
| 10      | Serving GPT-2 with MAX        | Connect the model to `max serve`; the custom architecture pattern |

## Project structure

```text
max-llm-book/
├── book/                  # mdBook tutorial documentation
│   └── src/
│       ├── introduction.md
│       ├── serve_first.md
│       ├── step_01.md ... step_10.md
│       └── SUMMARY.md
├── gpt2.py               # Complete GPT-2 implementation
├── gpt2_arch/            # Custom architecture package for `max serve`
├── tests/                # Tests for gpt2.py and gpt2_arch/
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

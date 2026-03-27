# Build an LLM from scratch with MAX

A guided tour of a complete GPT-2 implementation using Modular's MAX platform.
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

```bash
pixi run gpt2
```

This downloads the pretrained GPT-2 weights from HuggingFace, compiles the
model, and starts an interactive prompt where you can enter text and see
generated completions.

Additional modes:

```bash
pixi run gpt2 -- --prompt "Once upon a time"   # single generation, then exit
pixi run gpt2 -- --chat                         # streaming multi-turn chat
pixi run gpt2 -- --benchmark                    # tokens/sec benchmark
```

### Read the book

```bash
pixi run book
```

Or read it online at [llm.modular.com](https://llm.modular.com/).

## What the book covers

The tutorial walks through `gpt2.py` section by section:

| Section | Topic                       | What you'll learn                                          |
|---------|-----------------------------|------------------------------------------------------------|
| 1       | Model configuration         | Architecture hyperparameters and HuggingFace compatibility |
| 2       | Feed-forward network        | Two-layer MLP with GELU activation                         |
| 3       | Causal masking              | Preventing attention to future tokens                      |
| 4       | Multi-head attention        | Parallel attention across 12 heads                         |
| 5       | Layer normalization         | Pre-norm pattern for stable activations                    |
| 6       | Transformer block           | Residual connections and component wiring                  |
| 7       | Stacking transformer blocks | Embeddings and the 12-layer model body                     |
| 8       | Language model head         | Projecting hidden states to vocabulary logits              |
| 9       | Encode and decode tokens    | BPE tokenization with HuggingFace                          |
| 10      | Text generation             | Compiled sampling heads and Gumbel-max sampling            |
| 11      | Load weights and run model  | Lazy init, weight transposition, and model compilation     |
| 12      | Streaming chat              | Stop sequences, BPE boundary handling, and live rendering  |

## Project structure

```text
max-llm-book/
├── book/                  # mdBook tutorial documentation
│   └── src/
│       ├── introduction.md
│       ├── step_01.md ... step_12.md
│       └── SUMMARY.md
├── gpt2.py               # Complete GPT-2 implementation
├── tests/                # Tests for gpt2.py
├── pixi.toml             # Project dependencies and tasks
└── README.md             # This file
```

## Learning resources

- **MAX Documentation**: [docs.modular.com](https://docs.modular.com/)
- **HuggingFace GPT-2**: [huggingface.co/gpt2](https://huggingface.co/gpt2)
- **Attention Is All You Need**:
  [arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
- **Language Models are Unsupervised Multitask Learners** (GPT-2 paper):
  [openai.com](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

## Contributing

Found an issue or want to improve the tutorial? Contributions welcome:

1. File issues for bugs or unclear explanations
2. Suggest improvements to code examples or visualizations
3. Open a pull request with fixes or additions

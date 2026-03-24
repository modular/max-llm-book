# Project Setup

Clone [the GitHub repository](https://github.com/modular/max-llm-book) and
navigate to it:

```sh
git clone https://github.com/modular/max-llm-book
cd max-llm-book
```

Then download and install [pixi](https://pixi.sh/dev/):

```sh
curl -fsSL https://pixi.sh/install.sh | sh
```

## Run the model

To run the complete GPT-2 implementation interactively:

```bash
pixi run gpt2
```

This loads the pretrained GPT-2 weights from Hugging Face and starts an
interactive prompt. Enter any text and the model will generate a continuation.

You can also run the model with a single prompt and exit:

```bash
pixi run gpt2 -- --prompt "The quick brown fox"
```

Or open the streaming chat interface:

```bash
pixi run gpt2 -- --chat
```

## How to read this book

The tutorial walks through `gpt2.py`, the complete GPT-2 implementation. Each
section explains one component of the model, shows the relevant code snippet,
and explains how it works and why it's designed that way.

You don't need to write any code. Read each section, follow along in the source
file if you like, and run the model to see the output.

## A note on compile times

Compile times are actively being improved. As MAX continues to evolve, you
should expect performance improvements alongside upcoming Modular releases.

## Using code assistants

Code assistants like [Claude](https://claude.ai), [Cursor](https://cursor.sh),
or similar tools can help you navigate this tutorial. They're particularly
useful for:

- **Explaining concepts**: Ask about transformer architecture, attention
  mechanisms, or any component in the tutorial
- **Understanding the MAX API**: Get clarification on MAX Framework methods,
  parameters, and patterns
- **Exploring alternatives**: Ask "why this approach?" to deepen your
  understanding

If you're using Claude, see
[claude.md](https://github.com/modular/max-llm-book/blob/main/book/src/claude.md)
for custom instructions tailored to this tutorial.

## Prerequisites

This tutorial assumes:

- **Basic Python knowledge**: Classes, functions, type hints
- **Familiarity with neural networks**: What embeddings and layers do (we'll
  explain the specifics)
- **Interest in understanding**: Curiosity matters more than prior transformer
  experience

You'll need to meet the [system requirements](https://docs.modular.com/max/packages#system-requirements).

Ready? Start with [Section 1: Model configuration](./step_01.md).

# Set up the project

This page walks you through cloning the repository and installing the
dependencies you need to follow the tutorial.

## Prerequisites

This tutorial assumes the following:

- **Basic Python knowledge**: Classes, functions, type hints.
- **Familiarity with neural networks**: What embeddings and layers do (we'll
  explain the specifics).

Check the
[MAX system requirements](https://docs.modular.com/max/packages#system-requirements)
to confirm your platform is supported before continuing.

## Install

Clone [the GitHub repository](https://github.com/modular/max-llm-book) and
navigate to it:

```sh
git clone https://github.com/modular/max-llm-book
cd max-llm-book
```

Install [pixi](https://pixi.sh/dev/):

```sh
curl -fsSL https://pixi.sh/install.sh | sh
```

Install the tutorial's dependencies:

```sh
pixi install
```

**Next**: [Run the model](./serve_first.md) serves GPT-2 and calls the
endpoint before you write a line of model code.

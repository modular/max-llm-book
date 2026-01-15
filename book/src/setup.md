# Project Setup

You'll first need to clone [the GitHub repository](https://github.com/modular/max-llm-book) and navigate to the repository:

```sh
git clone https://github.com/modular/max-llm-book
cd max-llm-book
```

Then download and install [pixi](https://pixi.sh/dev/):

```
curl -fsSL https://pixi.sh/install.sh | sh
```

## How to use the book

To validate a step, use the corresponding check command. For example, to check
Step 01:

```bash
pixi run s01
```

Each step includes automated checks that verify your implementation before moving
forward. This immediate feedback helps you catch issues early and build
confidence. Initially, checks will fail because the implementation isn't complete:

```sh
✨ Pixi task (s01): python checks/check_step_01.py
Running checks for Step 01: Model Configuration...

✅ GPT2Config can be instantiated with default values

❌ ERRORS:
  - GPT2Config must be a dataclass (use @dataclass decorator)
  - Field 'vocab_size' has incorrect value: expected 50257, got None
  - Field 'n_positions' has incorrect value: expected 1024, got None
# ...
```

Each failure tells you exactly what to implement.

When your implementation is
correct, you'll see:

```output
✨ Pixi task (s01): python checks/check_step_01.py
Running checks for Step 01: Model Configuration...

✅ GPT2Config is a dataclass
✅ GPT2Config can be instantiated with default values
✅ vocab_size = 50257
✅ n_positions = 1024
# ...
```

The check output tells you exactly what needs to be fixed, making it easy to
iterate until your implementation is correct. Once all checks pass, you're ready
to move on to the next step.

## A note on compile times

Compile times are actively being improved. As MAX continues to evolve, you
should expect performance improvements alongside upcoming Modular releases.

## Using code assistants

Code assistants like [Claude](https://claude.ai), [Cursor](https://cursor.sh),
or similar tools can help you navigate this tutorial. They're particularly
useful for:

- **Explaining concepts**: Ask about transformer architecture, attention
  mechanisms, or any step in the tutorial
- **Understanding the MAX API**: Get clarification on MAX Framework methods,
  parameters, and patterns
- **Debugging check failures**: Paste check output to understand what's missing
- **Exploring alternatives**: Ask "why this approach?" to deepen your understanding

If you're using Claude, see [claude.md](./claude.md) for custom instructions
tailored to this tutorial.

## Prerequisites

This tutorial assumes:

- **Basic Python knowledge**: Classes, functions, type hints
- **Familiarity with neural networks**: What embeddings and layers do (we'll
  explain the specifics)
- **Interest in understanding**: Curiosity matters more than prior transformer
  experience

Whether you're exploring MAX for the first time or deepening your understanding
of model architecture, this tutorial provides hands-on experience you can apply
to current projects and learning priorities.

Ready to build? Let's get started with
[Step 01: Model configuration](./step_01.md).

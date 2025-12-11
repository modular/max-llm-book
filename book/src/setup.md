# How this book works

Each step includes automated tests that verify your implementation before moving
forward. This immediate feedback helps you catch issues early and build
confidence.

You'll first need to clone [the GitHub repository](https://github.com/modular/max-llm-book) and navigate to the repository:

```sh
git clone https://github.com/modular/max-llm-book
cd max-llm-book
```

Then download and install [pixi](https://pixi.sh/dev/):

```
curl -fsSL https://pixi.sh/install.sh | sh
```

To validate a step, use the corresponding test command. For example, to test
Step 01:

```bash
pixi run s01
```

Initially, tests will fail because the implementation isn't complete:

```sh
✨ Pixi task (s01): python tests/test.step_01.py
Running tests for Step 01: Create Model Configuration...

Results:
❌ dataclass is not imported from dataclasses
❌ GPT2Config does not have the @dataclass decorator
❌ vocab_size is incorrect: expected match with Hugging Face model configuration, got None
# ...
```

Each failure tells you exactly what to implement.

When your implementation is
correct, you'll see:

```output
✨ Pixi task (s01): python tests/test.step_01.py
Running tests for Step 01: Create Model Configuration...

Results:
✅ dataclass is correctly imported from dataclasses
✅ GPT2Config has the @dataclass decorator
✅ vocab_size is correct
# ...
```

The test output tells you exactly what needs to be fixed, making it easy to
iterate until your implementation is correct. Once all checks pass, you're ready
to move on to the next step.

## A note on compile times

Compile times are actively being improved. As MAX continues to evolve, you
should expect performance improvements alongside upcoming Modular releases.

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

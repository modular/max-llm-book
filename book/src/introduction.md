# Introduction

Transformer models power today's most impactful AI applications—from language
models like ChatGPT to code generation tools like GitHub Copilot. Maybe you've
been asked to adapt one of these models for your team, or you want to understand
what's actually happening when you call an inference API. Either way, building a
transformer from scratch is one of the best ways to truly understand how they
work.

This guide walks you through implementing GPT-2 using Modular's MAX framework.
You'll build each component yourself—embeddings, attention mechanisms,
feed-forward layers—and see how they fit together into a complete language
model. By the end, you'll be able to adapt models to your specific needs, debug
performance issues by understanding what's happening under the hood, and make
informed architecture decisions when designing ML systems.

> **Learning by building**: This tutorial follows a format popularized by Andrej
> Karpathy's educational work and Sebastian Raschka's hands-on approach. Rather
> than abstract theory, you'll implement each component yourself, building
> intuition through practice.

## Why MAX?

Traditional ML development often feels like stitching together tools that
weren't designed to work together. You write your model in PyTorch, optimize in
CUDA, convert to ONNX for deployment, then use separate serving tools. Each
handoff introduces complexity.

MAX Framework takes a different approach: everything happens in one unified
system. You write Python code to define your model, load weights, and run
inference—all in MAX's Python API. The Engine handles optimization
automatically, while MAX Serve manages deployment. No context switching, no
incompatible toolchains.

When you build GPT-2 in this guide, you'll load pretrained weights from
HuggingFace, implement the architecture, and run text generation—all in the same
environment. The skills transfer directly to building custom architectures. Once
you understand how GPT-2's components fit together, you can mix and match these
patterns for whatever model you need.

## Why Puzzles?

This tutorial emphasizes **active problem-solving over passive reading**. Each
step presents a focused implementation task with:

1. **Clear context**: What you're building and why it matters
2. **Guided implementation**: Code structure with specific tasks to complete
3. **Immediate validation**: Tests that verify correctness before moving forward
4. **Conceptual grounding**: Explanations that connect code to architecture

Rather than presenting complete solutions, this approach helps you develop
intuition for **when** and **why** to use specific patterns. The skills you
build extend beyond GPT-2 to model development more broadly.

You can work through the tutorial sequentially for comprehensive understanding,
or skip directly to topics you need. Each step is self-contained enough to be
useful independently while building toward a complete implementation.

## What you'll build

This tutorial guides you through building GPT-2 in manageable steps:

| Step | Component                         | What you'll learn                                                 |
|------|-----------------------------------|-------------------------------------------------------------------|
| 1    | Model configuration               | Define architecture hyperparameters matching HuggingFace GPT-2    |
| 2    | Causal masking                    | Create attention masks to prevent looking at future tokens        |
| 3    | Layer normalization               | Stabilize activations for effective training                      |
| 4    | GPT-2 MLP (feed-forward network)  | Build the position-wise feed-forward network with GELU activation |
| 5    | Token embeddings                  | Convert token IDs to continuous vector representations            |
| 6    | Position embeddings               | Encode sequence order information                                 |
| 7    | Query/Key/Value projections       | Transform embeddings for attention computation (single head)      |
| 8    | Attention mechanism               | Implement scaled dot-product attention with causal masking        |
| 9    | Multi-head attention              | Extend to multiple parallel attention heads                       |
| 10   | Residual connections & layer norm | Enable training deep networks with skip connections               |
| 11   | Transformer block                 | Combine attention and MLP into the core building block            |
| 12   | Stacking transformer blocks       | Create the complete 12-layer GPT-2 model                          |
| 13   | Language model head               | Project hidden states to vocabulary logits                        |
| 14   | Text generation                   | Generate text autoregressively with temperature sampling          |

Each step includes:

- Conceptual explanation of the component's role
- Implementation tasks with inline guidance
- Validation tests that verify correctness
- Connections to broader model development patterns

By the end, you'll have a complete GPT-2 implementation and practical experience
with MAX's Python API—skills you can immediately apply to your own projects.

## How This Works

Each step includes automated tests that verify your implementation before moving
forward. This immediate feedback helps you catch issues early and build
confidence.

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

Each failure tells you exactly what to implement. When your implementation is
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

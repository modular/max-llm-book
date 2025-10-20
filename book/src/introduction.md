# Introduction

Transformer models power today's most impactful AI applications—from language
models like ChatGPT to code generation tools like GitHub Copilot. Understanding
how these architectures work isn't just academic curiosity; it's a practical
skill that enables you to:

- **Adapt models** to your specific use cases and constraints
- **Debug performance issues** by understanding what's happening under the hood
- **Make informed architecture decisions** when designing ML systems
- **Optimize deployment** by knowing which components matter most

GPT-2, released by OpenAI in 2019, remains an excellent learning vehicle. It's
large enough to demonstrate real transformer architecture patterns, yet small
enough to understand completely. Every modern language model—from GPT-4 to
Llama—builds on these same fundamental components.

> **Learning by building**: This tutorial follows a format popularized by Andrej
> Karpathy's educational work and Sebastian Raschka's hands-on approach. Rather
> than abstract theory, you'll implement each component yourself, building
> intuition through practice.

## Why MAX?

The Modular Platform accelerates AI inference and abstracts hardware complexity
to make AI development faster and more portable. Unlike traditional ML
frameworks that evolved organically over time, MAX was built from the ground up
to address modern AI development challenges.

By implementing GPT-2 in MAX, you'll learn not just transformer architecture,
but also how MAX represents and optimizes neural networks. These skills transfer
directly to building your own custom architecture. 

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

## What You'll Build

This tutorial guides you through building GPT-2 in manageable steps:

| Step | Component             | What You'll Learn                                                                    |
|------|-----------------------|--------------------------------------------------------------------------------------|
| 1    | Model Configuration   | Define architecture hyperparameters and ensure compatibility with pretrained weights |
| 2    | Token Embeddings      | Convert token IDs to continuous vector representations                               |
| 3    | Position Embeddings   | Encode sequence order information                                                    |
| 4    | Layer Normalization   | Stabilize activations for effective training                                         |
| ...  | Additional components | (Coming soon)                                                                        |

Each step includes:

- Conceptual explanation of the component's role
- Implementation tasks with inline guidance
- Validation tests that verify correctness
- Connections to broader model development patterns

By the end, you'll have a complete GPT-2 implementation and practical experience
with MAX's Graph API—skills you can immediately apply to your own projects.

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

Ready to build? Let's get started with [Step 01: Model Configuration](./step_01.md).

# Build an LLM from scratch in MAX

<div class="warning">

**Experimental APIs:** The APIs in the
[experimental](https://docs.modular.com/max/api/python/experimental/) package
are subject to change. Share feedback on the
[MAX LLMs forum](https://forum.modular.com/tag/max-llms).

</div>

Transformer models power today's most impactful AI applications, from language models like ChatGPT to code generation tools like GitHub Copilot. Maybe you've been asked to adapt one of these models for your team, or you want to understand what's actually happening when you call an inference API. Either way, building a transformer from scratch is one of the best ways to truly understand how they work.

This guide walks you through implementing GPT-2 using Modular's MAX framework [experimental API](https://docs.modular.com/max/api/python/experimental/). You'll build each component yourself: embeddings, attention mechanisms, and feed-forward layers. You'll see how they fit together into a complete language model by completing the sequential coding challenges in the tutorial [GitHub repository](https://github.com/modular/max-llm-book).

<div class="note">

<strong>This API is unstable</strong>: This tutorial is built on the MAX
Experimental API, which we expect to change over time and expand to include new
features and functionality. As it evolves, we plan to update the tutorial
accordingly. When this API comes out of experimental development, the tutorial
content will also enter a more stable state. While in development, this tutorial
will be pinned to a major release version.

</div>

## Why GPT-2?

It's the architectural foundation for modern language models. LLaMA, Mistral, GPT-4; they're all built on the same core components you'll implement here:

- multi-head attention
- feed-forward layers
- layer normalization

Modern variants add refinements like grouped-query attention or mixture of experts, but the fundamentals remain the same. GPT-2 is complex enough to teach real transformer architecture but simple enough to implement completely and understand deeply. When you grasp how its pieces fit together, you understand how to build any transformer-based model.

> **Learning by building**: This tutorial follows a format popularized by Andrej
> Karpathy's educational work and Sebastian Raschka's hands-on approach. Rather
> than abstract theory, you'll implement each component yourself, building
> intuition through practice.

## Why MAX?

Traditional ML development often feels like stitching together tools that
weren't designed to work together. Maybe you write your model in PyTorch, optimize in
CUDA, convert to ONNX for deployment, then use separate serving tools. Each
handoff introduces complexity.

MAX Framework takes a different approach: everything happens in one unified
system. You write code to define your model, load weights, and run inference,
all in MAX's Python API. The MAX Framework handles optimization automatically and
you can even use MAX Serve to manage your deployment.

When you build GPT-2 in this guide, you'll load pretrained weights from
Hugging Face, implement the architecture, and run text generation, all in the same
environment.

## Why coding challenges?

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

| Step | Component                                    | What you'll learn                                                  |
|------|----------------------------------------------|--------------------------------------------------------------------|
| 1    | [Model configuration](./step_01.md)          | Define architecture hyperparameters matching HuggingFace GPT-2.    |
| 2    | [Feed-forward network](./step_02.md)         | Build the position-wise feed-forward network with GELU activation. |
| 3    | [Causal masking](./step_03.md)               | Create attention masks to prevent looking at future tokens.        |
| 4    | [Multi-head attention](./step_04.md)         | Implement scaled dot-product attention with multiple heads.        |
| 5    | [Layer normalization](./step_05.md)          | Ensure activation values are within a stable range.                      |
| 6    | [Transformer block](./step_06.md)            | Combine attention and MLP with residual connections.               |
| 7    | [Stacking transformer blocks](./step_07.md)  | Create the complete 12-layer GPT-2 model with embeddings.          |
| 8    | [Language model head](./step_08.md)          | Project hidden states to vocabulary logits.                        |
| 9    | [Encode and decode tokens](./step_09.md)     | Convert between text and token IDs using HuggingFace tokenizer.    |
| 10   | [Text generation](./step_10.md)              | Generate text autoregressively with temperature sampling.          |
| 11   | [Load weights and run model](./step_11.md)   | Load pretrained weights and interact with your complete model.     |

By the end, you'll have a complete GPT-2 implementation and practical experience
with MAX's Python API. These are skills you can immediately apply to your own projects.

> **Note on training vs. inference**: While some steps reference concepts from
> training (like layer normalization for "stabilizing activations"), this
> tutorial focuses on inference using pretrained weights from Hugging Face.
> Training is not in scope, but we include these architectural details for
> learning purposes and completeness—understanding why each layer exists helps
> you reason about model behavior and adapt architectures for your own needs.

## Try it first

Before diving into the implementation, you can experience what you'll build by running the complete reference model:

```bash
pixi run main
```

This runs the complete GPT-2 implementation from [`main.py`](https://github.com/modular/max-llm-book/blob/main/main.py), loading pretrained weights and starting an interactive prompt where you can enter text and see the model generate completions. It's the same model you'll build step-by-step through the tutorial.

When you've completed every step of the tutorial, you can run your own implementation the exact same way:

```bash
pixi run gpt2
```

This runs your completed `steps/step_11.py`, demonstrating that your implementation works identically to the reference. Both commands load the same pretrained weights, compile the model, and provide an interactive generation experience.

## Get started

To install the tutorial and begin building, follow the steps in
[Setup](./setup.md).

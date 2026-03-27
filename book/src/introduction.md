# Build an LLM from scratch in MAX

Transformer models power today's most impactful AI applications, from language
models like ChatGPT to code generation tools like GitHub Copilot. Maybe you've
been asked to adapt one of these models for your team, or you want to understand
what's actually happening when you call an inference API. Either way, building
a transformer from scratch is one of the best ways to truly understand how they
work.

This guide walks you through a complete GPT-2 implementation using the
[MAX Python API](https://docs.modular.com/max/api/python/). Each section
explains a component of the model—embeddings, attention mechanisms, feed-forward
layers—and shows exactly how it's implemented in the
[GitHub repository](https://github.com/modular/max-llm-book).

## Why GPT-2?

It's the architectural foundation for modern language models. LLaMA, Mistral,
GPT-4; they're all built on the same core components you'll find here:

- multi-head attention
- feed-forward layers
- layer normalization

Modern variants add refinements like grouped-query attention or mixture of
experts, but the fundamentals remain the same. GPT-2 is complex enough to teach
real transformer architecture but simple enough to implement completely and
understand deeply. When you grasp how its pieces fit together, you understand
how to build any transformer-based model.

> **Learning by example**: Rather than abstract theory, this tutorial walks
> through a complete, working implementation and explains how each component
> works and why it's designed that way.

## Why MAX?

Traditional ML development often feels like stitching together tools that
weren't designed to work together. Maybe you write your model in PyTorch,
optimize in CUDA, convert to ONNX for deployment, then use separate serving
tools. Each handoff introduces complexity.

MAX Framework takes a different approach: everything happens in one unified
system. You write code to define your model, load weights, and run inference,
all in MAX's Python API. The MAX Framework handles optimization automatically
and you can even use MAX Serve to manage your deployment.

The GPT-2 implementation in this guide loads pretrained weights from Hugging
Face, implements the architecture, and runs text generation, all in the same
environment.

## What you'll explore

Each section explains a component of the model through the working code in
`gpt2.py`:

| Section | Component                                   | What you'll learn                                                  |
|---------|---------------------------------------------|--------------------------------------------------------------------|
| 1       | [Model configuration](./step_01.md)         | Define architecture hyperparameters matching HuggingFace GPT-2.    |
| 2       | [Feed-forward network](./step_02.md)        | Build the position-wise feed-forward network with GELU activation. |
| 3       | [Causal masking](./step_03.md)              | Create attention masks to prevent looking at future tokens.        |
| 4       | [Multi-head attention](./step_04.md)        | Implement scaled dot-product attention with multiple heads.        |
| 5       | [Layer normalization](./step_05.md)         | Ensure activation values are within a stable range.                |
| 6       | [Transformer block](./step_06.md)           | Combine attention and MLP with residual connections.               |
| 7       | [Stacking transformer blocks](./step_07.md) | Create the complete 12-layer GPT-2 model with embeddings.          |
| 8       | [Language model head](./step_08.md)         | Project hidden states to vocabulary logits.                        |
| 9       | [Encode and decode tokens](./step_09.md)    | Convert between text and token IDs using HuggingFace tokenizer.    |
| 10      | [Text generation](./step_10.md)             | Generate text autoregressively with temperature sampling.          |
| 11      | [Load weights and run model](./step_11.md)  | Load pretrained weights and interact with your complete model.     |
| 12      | [Streaming chat](./step_12.md)              | Build a streaming multi-turn chat interface using stop sequences.  |

By the end, you'll understand every line of a complete GPT-2 implementation and
have practical experience with MAX's Python API—skills you can apply directly to
your own projects.

> **Note on training vs. inference**: This tutorial focuses on inference using
> pretrained weights from Hugging Face. Training is not in scope, but we
> include architectural details like layer normalization for completeness—
> understanding why each layer exists helps you reason about model behavior and
> adapt architectures for your own needs.

## Try it now

Before reading the implementation, run the complete GPT-2 model:

```bash
pixi run gpt2
```

This loads the pretrained weights and starts an interactive prompt where you can
enter text and see the model generate completions. It's the same model you'll
read through step-by-step in the tutorial.

## Get started

To install the project and begin, follow the steps in [Setup](./setup.md).

# Build an LLM from scratch in MAX

Transformer models power today's most impactful AI applications, from language
models like ChatGPT to code generation tools like GitHub Copilot. If you want
to understand what's actually happening when you call an inference API, or you
need to adapt one of these models for your own work, building one from scratch
is the fastest path to that understanding.

This guide walks you through a complete GPT-2 implementation using the
[MAX Python API](https://docs.modular.com/max/api/python/). You'll start by
running a working model, then build it component by component: embeddings,
attention, feed-forward layers, and the serving layer that connects it all.
Everything runs from the
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

## Why MAX?

Building a model for inference typically means a separate tool for each stage:
one framework for model definition, another for optimization, another for
serving. Each handoff is another thing to learn and another place for things
to break.

MAX handles all of it in one Python API. You define the model, load weights,
run inference, and serve with `max serve`, all in the same environment.

The GPT-2 implementation in this guide loads pretrained weights from Hugging
Face, implements the architecture, and serves text generation, all in that same
environment.

## How to read this book

You'll start by serving GPT-2 with MAX, getting a working model running before
you write a line of model code. From there, you'll build the transformer block
component by component, assemble those components into the full model, and
finish by learning how inference and serving work with `max serve`.

| Chapter                                   | What you'll learn                                                  |
|-------------------------------------------|--------------------------------------------------------------------|
| [Project setup](./setup.md)               | Install MAX and clone the repository                               |
| [Run the model](./serve_first.md)         | Serve GPT-2 and call the endpoint                                  |
| [Model configuration](./step_01.md)       | Define the architecture hyperparameters                            |
| [Feed-forward network](./step_02.md)      | Build the position-wise MLP with GELU activation                   |
| [Causal masking](./step_03.md)            | Create attention masks that prevent looking at future tokens       |
| [Multi-head attention](./step_04.md)      | Implement scaled dot-product attention with multiple heads         |
| [Layer normalization](./step_05.md)       | Stabilize activations between sub-layers                           |
| [Transformer block](./step_06.md)         | Combine attention and MLP with residual connections                |
| [Stack transformer blocks](./step_07.md)  | Build the complete 12-layer GPT-2 with embeddings                  |
| [Language model head](./step_08.md)       | Project hidden states to vocabulary logits                         |
| [Weight adaptation](./step_09.md)         | Reconcile GPT-2's Hugging Face checkpoint with MAX's weight layout |
| [KV cache configuration](./step_10.md)    | Expose attention dimensions for cache pre-allocation               |
| [Pipeline model](./step_11.md)            | Load, compile, and execute the model inside `max serve`            |
| [Architecture registration](./step_12.md) | Declare the package to `max serve` and wire all pieces together    |

The code is pre-written in the repository. Training is not in scope.

Start with [Project setup](./setup.md) to get your environment ready.

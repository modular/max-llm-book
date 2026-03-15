# Model configuration

<div class="note">

Define the GPT-2 model architecture parameters using a configuration class.

</div>

Before implementing GPT-2, you need to define its architecture: the dimensions,
layer counts, and structural parameters that determine how the model processes
information.

`GPT2Config` holds all the architectural decisions for GPT-2—embedding
dimensions, number of transformer layers, number of attention heads. These
parameters define the shape and capacity of the model.

OpenAI trained the original GPT-2 model with specific parameters available in
the
[config.json file](https://huggingface.co/openai-community/gpt2/blob/main/config.json)
on Hugging Face. Using the exact same values lets us load OpenAI's pretrained
weights in the final step.

## The configuration parameters

Each field controls a different aspect of the model:

- `vocab_size`: Size of the token vocabulary (50,257). This number is
  50,000 Byte Pair Encoding tokens + 256 byte-level tokens (fallback for rare
  characters) + 1 special token.
- `n_positions`: Maximum sequence length, also called the context window
  (1,024). Longer sequences require quadratic memory in attention.
- `n_embd`: Embedding dimension, the size of the hidden states that flow
  through the model (768). This determines the model's capacity to represent
  information.
- `n_layer`: Number of transformer blocks stacked vertically (12). More
  layers allow the model to learn more complex patterns.
- `n_head`: Number of attention heads per layer (12). Multiple heads let the
  model attend to different types of patterns simultaneously.
- `n_inner`: Dimension of the MLP intermediate layer (optional, defaults to
  4× embedding). The 4× ratio comes from the original
  [_Attention is all you need_](https://arxiv.org/abs/1706.03742) paper.
- `layer_norm_epsilon`: Small constant for numerical stability in layer
  normalization (1e-5). Prevents division by zero when variance is very small.

These values define the _small_ GPT-2 model. OpenAI released four sizes (small,
medium, large, XL), each scaling these parameters up.

## The code

Python's
[`@dataclass`](https://docs.python.org/3/library/dataclasses.html) decorator
eliminates boilerplate. Instead of writing `__init__` manually, you declare
fields with type hints and default values:

```python
{{#include ../../gpt2.py:model_configuration}}
```

The `n_inner: int | None = None` field is optional. When `None`, the
transformer block defaults to 4× the embedding dimension (3,072). This lets you
override the inner dimension for experimental architectures without changing the
other components.

**Next**: [Section 2](./step_02.md) implements the feed-forward network—the MLP
that processes information after attention in each transformer block.

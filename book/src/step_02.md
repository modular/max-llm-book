# Feed-forward network (MLP)

<div class="note">

Learn to build the feed-forward network—also known as a multilayer perceptron (MLP)—that processes information after attention in each transformer block.

</div>

In this step, you'll create the `GPT2MLP` class: a two-layer feed-forward
network that appears after attention in every transformer block. The MLP expands
the embedding dimension by 4× (768 → 3,072), applies GELU activation for
non-linearity, then projects back to the original dimension.

While attention lets tokens communicate with each other, the MLP processes each
position independently. Attention aggregates information through weighted sums
(linear operations), but the MLP adds non-linearity through GELU activation.
This combination allows the model to learn complex patterns beyond what linear
transformations alone can capture.

GPT-2 uses a 4× expansion ratio (768 to 3,072 dimensions) because this was found
to work well in the original Transformer paper and has been validated across
many architectures since.

## Understanding the components

The MLP has three steps applied in sequence:

**Expansion layer (`c_fc`)**: Projects from 768 to 3,072 dimensions using a linear layer. This expansion gives the network more capacity to process information.

**GELU activation**: Applies Gaussian Error Linear Unit, a smooth non-linear
function. GPT-2 uses `approximate="tanh"` for the tanh-based approximation
instead of the exact computation. This approximation was faster when GPT-2 was
implemented, but while exact GELU is fast enough now, we use the approximation
to match the original weights.

**Projection layer (`c_proj`)**: Projects back from 3,072 to 768 dimensions
using another linear layer. This returns to the embedding dimension so outputs
can be added to residual connections.

The layer names `c_fc` (fully connected) and `c_proj` (projection) match Hugging
Face's GPT-2 checkpoint structure. This naming is essential for loading
pretrained weights.

<div class="note">

<div class="title">MAX operations</div>

You'll use the following MAX operations to complete this task:

**Linear layers**:

- [`Linear(in_features, out_features, bias=True)`](https://docs.modular.com/max/api/python/nn/module_v3#max.nn.module_v3.Linear): Applies linear transformation `y = xW^T + b`

**GELU activation**:

- [`F.gelu(input, approximate="tanh")`](https://docs.modular.com/max/api/python/experimental/functional#max.experimental.functional.gelu): Applies GELU activation with tanh approximation for faster computation

</div>

## Implementing the MLP

You'll create the `GPT2MLP` class that chains two linear layers with GELU
activation between them. The implementation is straightforward - three
operations applied in sequence.

First, import the required modules. You'll need `functional as F` for the GELU
activation, `Tensor` for type hints, `Linear` for the layers, and `Module` as
the base class.

In the `__init__` method, create two linear layers:

- Expansion layer: `Linear(embed_dim, intermediate_size, bias=True)` stored as
  `self.c_fc`
- Projection layer: `Linear(intermediate_size, embed_dim, bias=True)` stored as
  `self.c_proj`

Both layers include bias terms (`bias=True`). The intermediate size is typically
4× the embedding dimension.

In the `forward` method, apply the three transformations:

1. Expand: `hidden_states = self.c_fc(hidden_states)`
2. Activate: `hidden_states = F.gelu(hidden_states, approximate="tanh")`
3. Project: `hidden_states = self.c_proj(hidden_states)`

Return the final `hidden_states`. The input and output shapes are the same:
`[batch, seq_length, embed_dim]`.

**Implementation** (`step_02.py`):

```python
{{#include ../../steps/step_02.py}}
```

### Validation

Run `pixi run s02` to verify your implementation.

<details>
<summary>Show solution</summary>

```python
{{#include ../../solutions/solution_02.py}}
```

</details>

**Next**: In [Step 03](./step_03.md), you’ll implement causal masking to prevent tokens from attending to future positions in autoregressive generation.

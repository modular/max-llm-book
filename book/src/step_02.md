# Feed-forward network (MLP)

<div class="note">

Build the feed-forward network, also known as a multilayer perceptron (MLP),
that processes information after attention in each transformer block.

</div>

Every transformer block contains a two-layer feed-forward network. `GPT2MLP`
expands the embedding dimension by 4× (768 → 3,072), applies GELU activation
for non-linearity, then projects back to the original dimension.

While attention lets tokens communicate with each other, the MLP processes each
position independently. Attention aggregates information through weighted sums
(linear operations), but the MLP adds non-linearity through GELU activation.
This combination allows the model to learn complex patterns beyond what linear
transformations alone can capture.

GPT-2 uses a 4× expansion ratio because this was found to work well in the
[original Transformer paper](https://arxiv.org/abs/1706.03762) and has been
validated across many architectures since.

## MLP operations

The MLP has three operations:

- **Expansion layer (`c_fc`)**: Projects from 768 to 3,072 dimensions, giving
  the network more capacity to process information.
- **GELU activation**: Applies Gaussian Error Linear Unit, a smooth non-linear
  function. GPT-2 uses `approximate="tanh"` for the tanh-based approximation,
  which was faster when GPT-2 was first implemented and is required here to
  match the original pretrained weights exactly.
- **Projection layer (`c_proj`)**: Projects back from 3,072 to 768 dimensions,
  returning to the embedding dimension so outputs can be added to residual
  connections.

The layer names `c_fc` (fully connected) and `c_proj` (projection) match Hugging
Face's GPT-2 checkpoint structure. This naming is essential for loading
pretrained weights in the final step.

## GPT2MLP

[`Linear(in_dim, out_dim, bias=True)`](https://docs.modular.com/max/api/python/generated/max.experimental.nn.Linear/)
applies `y = xW^T + b`. Both layers include bias terms.
[`F.gelu`](https://docs.modular.com/max/api/python/experimental.functional#max.experimental.functional.gelu)
applies the activation between them:

```python:gpt2.py
{{#include ../../gpt2_arch/gpt2.py:feed_forward_network}}
```

The input and output both have shape `[batch, seq_length, 768]`. The 3,072
intermediate dimension exists only inside the MLP; the transformer block sees
the same shape going in and coming out.

**Next**: [Causal masking](./step_03.md) prevents tokens
from attending to future positions during autoregressive generation.

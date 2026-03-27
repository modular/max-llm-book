# Feed-forward network (MLP)

<div class="note">

Build the feed-forward network—also known as a multilayer perceptron (MLP)—that
processes information after attention in each transformer block.

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
original Transformer paper and has been validated across many architectures
since.

## The three steps

**Expansion layer (`c_fc`)**: Projects from 768 to 3,072 dimensions. This
expansion gives the network more capacity to process information.

**GELU activation**: Applies Gaussian Error Linear Unit, a smooth non-linear
function. GPT-2 uses `approximate="tanh"` for the tanh-based approximation.
This approximation was faster when GPT-2 was first implemented, but we use it
here to match the original pretrained weights exactly.

**Projection layer (`c_proj`)**: Projects back from 3,072 to 768 dimensions.
This returns to the embedding dimension so outputs can be added to residual
connections.

The layer names `c_fc` (fully connected) and `c_proj` (projection) match Hugging
Face's GPT-2 checkpoint structure. This naming is essential for loading
pretrained weights in the final step.

## The code

[`Linear(in_features, out_features, bias=True)`](https://docs.modular.com/max/api/python/generated/max.nn.Linear)
applies `y = xW^T + b`. Both layers include bias terms.
[`F.gelu`](https://docs.modular.com/max/api/python/experimental.functional#max.experimental.functional.gelu)
applies the activation between them:

```python
{{#include ../../gpt2.py:feed_forward_network}}
```

The input and output both have shape `[batch, seq_length, 768]`. The 3,072
intermediate dimension exists only inside the MLP—the transformer block sees
the same shape going in and coming out.

**Next**: [Section 3](./step_03.md) implements causal masking to prevent tokens
from attending to future positions during autoregressive generation.

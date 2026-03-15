# Layer normalization

<div class="note">

Implement layer normalization to keep activations in a stable range throughout
the network.

</div>

`LayerNorm` normalizes activations across the feature dimension. For each input
position, it computes the mean and variance across all 768 features, normalizes
to zero mean and unit variance, then applies learned weight and bias parameters
to scale and shift the result.

Unlike batch normalization, [layer normalization](https://arxiv.org/abs/1607.06450)
works independently for each example, with no dependence on batch size and no
running statistics to track. This makes it ideal for transformers, where batch
sizes and sequence lengths vary.

GPT-2 applies layer normalization _before_ both the attention and MLP sublayers
in each transformer block (pre-normalization). This pattern stabilizes training
in deep networks by keeping activations in a consistent range as gradients flow
backward through 12 stacked blocks.

Layer normalization is required during inference too, not just training. The
pretrained weights were optimized assuming normalized inputs at each sublayer.
Skipping it would cause activations to be in completely different ranges than
what the model learned, producing poor or nonsensical output.

## The normalization formula

```math
output = weight * (x - mean) / sqrt(variance + epsilon) + bias
```

The mean and variance are computed across all features in each example.
`epsilon` (1e-5) prevents division by zero when variance is very small. The
learned `weight` scales the normalized result and `bias` shifts it—initialized
to ones and zeros so the initial transformation is identity.

## The code

[`F.layer_norm`](https://docs.modular.com/max/api/python/experimental.functional#max.experimental.functional.layer_norm)
computes the normalization and applies the learned parameters in one call. The
weight is initialized with
[`Tensor.ones`](https://docs.modular.com/max/api/python/generated/max.experimental.tensor.Tensor#max.experimental.tensor.Tensor.ones)
and the bias with
[`Tensor.zeros`](https://docs.modular.com/max/api/python/generated/max.experimental.tensor.Tensor#max.experimental.tensor.Tensor.zeros):

```python
{{#include ../../gpt2.py:layer_normalization}}
```

**Next**: [Section 6](./step_06.md) combines attention, MLP, layer
normalization, and residual connections into a complete transformer block.

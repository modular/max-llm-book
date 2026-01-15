# Layer normalization

<div class="note">

Learn to implement layer normalization for stabilizing neural network training.

</div>

In this step, you'll create the `LayerNorm` class that normalizes activations across the feature dimension. For each input, you compute the mean and variance across all features, normalize by subtracting the mean and dividing by the standard deviation, then apply learned weight and bias parameters to scale and shift the result.

Unlike batch normalization, [layer normalization](https://arxiv.org/abs/1607.06450) works independently for each example. This makes it ideal for transformers - no dependence on batch size, no tracking running statistics during inference, and consistent behavior between training and generation.

GPT-2 applies layer normalization before the attention and MLP blocks in each of its 12 transformer layers. This pre-normalization pattern stabilizes training in deep networks by keeping activations in a consistent range.

While layer normalization is most critical during training to stabilize gradients and prevent activations from exploding or vanishing, **it's still required during inference**. The pretrained GPT-2 model we're loading was trained with layer normalization - its learned weights and biases expect normalized inputs. Skipping layer normalization during inference would cause activations to be in completely different ranges than what the model learned during training, leading to poor or nonsensical outputs.

## Understanding the operation

Layer normalization normalizes across the feature dimension (the last dimension) independently for each example. It learns two parameters per feature: weight (gamma) for scaling and bias (beta) for shifting.

The normalization follows this formula:

```math
output = weight * (x - mean) / sqrt(variance + epsilon) + bias
```

The mean and variance are computed across all features in each example. After normalizing to zero mean and unit variance, the learned weight scales the result and the learned bias shifts it. The epsilon value (typically 1e-5) prevents division by zero when variance is very small.

<div class="note">

<div class="title">MAX operations</div>

You'll use the following MAX operations to complete this task:

**Modules**:

- [`Module`](https://docs.modular.com/max/api/python/nn/module_v3/): The Module class used for eager tensors

**Tensor initialization**:

- [`Tensor.ones()`](https://docs.modular.com/max/api/python/experimental/tensor#max.experimental.tensor.Tensor.ones): Creates tensor filled with 1.0 values
- [`Tensor.zeros()`](https://docs.modular.com/max/api/python/experimental/tensor#max.experimental.tensor.Tensor.zeros): Creates tensor filled with 0.0 values

**Layer normalization**:

- [`F.layer_norm()`](https://docs.modular.com/max/api/python/experimental/functional#max.experimental.functional.layer_norm): Applies layer normalization with parameters: `input`, `gamma` (weight), `beta` (bias), and `epsilon`

</div>

## Implementing layer normalization

You'll create the `LayerNorm` class that wraps MAX's layer normalization function with learnable parameters. The implementation is straightforward - two parameters and a single function call.

First, import the required modules. You'll need `functional as F` for the layer norm operation and `Tensor` for creating parameters.

In the `__init__` method, create two learnable parameters:

- Weight: `Tensor.ones([dim])` stored as `self.weight` - initialized to ones so the initial transformation is identity
- Bias: `Tensor.zeros([dim])` stored as `self.bias` - initialized to zeros so there's no initial shift

Store the epsilon value as `self.eps` for numerical stability.

In the `forward` method, apply layer normalization with `F.layer_norm(x, gamma=self.weight, beta=self.bias, epsilon=self.eps)`. This computes the normalization and applies the learned parameters in one operation.

**Implementation** (`step_05.py`):

```python
{{#include ../../steps/step_05.py}}
```

### Validation

Run `pixi run s05` to verify your implementation.

<details>
<summary>Show solution</summary>

```python
{{#include ../../solutions/solution_05.py}}
```

</details>

**Next**: In [Step 06](./step_06.md), you’ll combine multi-head attention, MLP, layer norm, and residual connections into a complete transformer block.

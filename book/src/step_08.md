# Step 08: Residual connections and layer normalization

<div class="note">

Learn to implement residual connections and layer normalization to enable
training deep transformer networks.

</div>

## Building the residual pattern

In this step, you'll combine residual connections and layer normalization into a
reusable pattern for transformer blocks. Residual connections add the input
directly to the output using `output = input + layer(input)`, creating shortcuts
that let gradients flow through deep networks. You'll implement this alongside
the layer normalization from Step 03.

GPT-2 uses pre-norm architecture where layer norm is applied before each
sublayer (attention or MLP). The pattern is `x = x + sublayer(layer_norm(x))`:
normalize first, process, then add the original input back. This is more stable
than post-norm alternatives for deep networks.

Residual connections solve the vanishing gradient problem. During
backpropagation, gradients flow through the identity path (`x = x + ...`)
without being multiplied by layer weights. This allows training networks with
12+ layers. Without residuals, gradients would diminish exponentially as they
propagate through many layers.

Layer normalization works identically during training and inference because it
normalizes each example independently. No batch statistics, no running averages,
just consistent normalization that keeps activation distributions stable
throughout training.

## Understanding the pattern

The pre-norm residual pattern combines three operations in sequence:

**Layer normalization**: Normalize the input with
`F.layer_norm(x, gamma=self.weight, beta=self.bias, epsilon=self.eps)`. This
uses learnable weight (gamma) and bias (beta) parameters to scale and shift the
normalized values. You already implemented this in Step 03.

**Sublayer processing**: Pass the normalized input through a sublayer (attention
or MLP). The sublayer transforms the data while the layer norm keeps its input
well-conditioned.

**Residual addition**: Add the original input back to the sublayer output using
simple element-wise addition: `x + sublayer_output`. Both tensors must have
identical shapes `[batch, seq_length, embed_dim]`.

The complete pattern is `x = x + sublayer(layer_norm(x))`. This differs from
post-norm `x = layer_norm(x + sublayer(x))`, as pre-norm is more stable because
normalization happens before potentially unstable sublayer operations.

<div class="note">

<div class="title">MAX operations</div>

You'll use the following MAX operations to complete this task:

**Layer normalization**:
- [`F.layer_norm(x, gamma, beta, epsilon)`](https://docs.modular.com/max/api/python/experimental/functional#max.experimental.functional.layer_norm): Normalizes across feature dimension

**Tensor initialization**:
- [`Tensor.ones([dim])`](https://docs.modular.com/max/api/python/experimental/tensor#max.experimental.tensor.Tensor.ones): Creates weight parameter
- [`Tensor.zeros([dim])`](https://docs.modular.com/max/api/python/experimental/tensor#max.experimental.tensor.Tensor.zeros): Creates bias parameter

</div>

## Implementing the pattern

You'll implement three classes that demonstrate the residual pattern:
`LayerNorm` for normalization, `ResidualBlock` that combines norm and residual
addition, and a standalone `apply_residual_connection` function.

First, import the required modules. You'll need `functional as F` for layer
norm, `Tensor` for parameters, `DimLike` for type hints, and `Module` as the
base class.

**LayerNorm implementation**:

In `__init__`, create the learnable parameters:
- Weight: `Tensor.ones([dim])` stored as `self.weight`
- Bias: `Tensor.zeros([dim])` stored as `self.bias`
- Store `eps` for numerical stability

In `forward`, apply normalization with
`F.layer_norm(x, gamma=self.weight, beta=self.bias, epsilon=self.eps)`. Returns
a normalized tensor with the same shape as input.

**ResidualBlock implementation**:

In `__init__`, create a `LayerNorm` instance:
`self.ln = LayerNorm(dim, eps=eps)`. This will normalize inputs before
sublayers.

In `forward`, implement the pre-norm pattern:
1. Normalize: `normalized = self.ln(x)`
2. Process: `sublayer_output = sublayer(normalized)`
3. Add residual: `return x + sublayer_output`

**Standalone function**:

Implement `apply_residual_connection(input_tensor, sublayer_output)` that
returns `input_tensor + sublayer_output`. This demonstrates the residual pattern
as a simple function.

**Implementation** (`step_08.py`):

```python
{{#include ../../steps/step_08.py}}
```

### Validation

Run `pixi run s08` to verify your implementation.

<details>
<summary>Show solution</summary>

```python
{{#include ../../solutions/solution_08.py}}
```

</details>

**Next**: In [Step 09](./step_09.md), you'll combine multi-head attention, MLP,
layer norm, and residual connections into a complete transformer block.

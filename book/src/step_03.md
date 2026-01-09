# Causal masking

<div class="note">

Learn to create attention masks to prevent the model from _seeing_ future tokens during [autoregressive](https://docs.modular.com/glossary/ai/autoregression) generation.

</div>

In this step you'll implement the `causal_mask()` function that's required for self-attention (the next step). This creates a [mask matrix](https://docs.modular.com/glossary/ai/attention-mask/) that prevents the model from _seeing_ future tokens when predicting the next token. The mask sets attention scores to negative infinity (`-inf`) for future positions. After softmax, these `-inf` values become zero probability, blocking information flow from later tokens.

<figure>
<img src="./images/causal-masking-light.png" alt="Causal mask matrix showing the lower triangular pattern where each token can only attend to previous tokens" class="light-mode-img" width="530" height="475">
<img src="./images/causal-masking-dark.png" alt="Causal mask matrix showing the lower triangular pattern where each token can only attend to previous tokens" class="dark-mode-img" width="530" height="475">
</figure>

GPT-2 generates text one token at a time, left-to-right. During training, causal masking prevents the model from "cheating" by looking ahead at tokens it should be predicting. Without this mask, the model would have access to information it won't have during actual text generation.

## Understanding the mask pattern

The mask creates a lower triangular pattern where each token can only attend to itself and previous tokens:

- Position 0 attends to: position 0 only
- Position 1 attends to: positions 0-1
- Position 2 attends to: positions 0-2
- And so on...

The mask shape is `(sequence_length, sequence_length + num_tokens)`. This shape is designed for [KV cache](https://docs.modular.com/glossary/ai/kv-cache/) compatibility during generation. The KV cache stores key and value tensors from previously generated tokens, so you only need to compute attention for new tokens while attending to both new tokens (sequence_length) and cached tokens (num_tokens). This significantly speeds up generation by avoiding recomputation.

<div class="note">

<div class="title">MAX operations</div>

You'll use the following MAX operations to complete this task:

**Functional decorator**:

- [`@F.functional`](https://docs.modular.com/max/api/python/experimental/functional/#max.experimental.functional.functional): Converts functions to graph operations for MAX compilation

**Tensor operations**:

- [`Tensor.constant()`](https://docs.modular.com/max/api/python/experimental/tensor#max.experimental.tensor.Tensor.constant): Creates a scalar constant tensor
- [`F.broadcast_to()`](https://docs.modular.com/max/api/python/experimental/functional#max.experimental.functional.broadcast_to): Expands tensor dimensions to target shape
- [`F.band_part()`](https://docs.modular.com/max/api/python/experimental/functional#max.experimental.functional.band_part): Extracts band matrix (keeps diagonal band, zeros out rest)

</div>

## Implementing the mask

You'll create the causal mask in several steps:

1. **Import required modules**:
   - [`Device`](https://docs.modular.com/max/api/python/driver) from `max.driver` - specifies hardware device (CPU/GPU)
   - [`DType`](https://docs.modular.com/max/api/python/dtype) from `max.dtype` - data type specification
   - [`functional`](https://docs.modular.com/max/api/python/experimental/functional) as `F` from `max.experimental` - functional operations library
   - [`Tensor`](https://docs.modular.com/max/api/python/experimental/tensor) from `max.experimental.tensor` - tensor operations
   - [`Dim`](https://docs.modular.com/max/api/python/graph/dim/#max.graph.dim.Dim) from `graph.dim` - dimension handling

2. **Add @F.functional decorator**: This converts the function to a MAX graph operation.

3. **Calculate total sequence length**: Combine `sequence_length` and `num_tokens` using `Dim()` to determine mask width.

4. **Create constant tensor**: Use `Tensor.constant(float("-inf"), dtype=dtype, device=device)` to create a scalar that will be broadcast.

5. **Broadcast to target shape**: Use `F.broadcast_to(mask, shape=(sequence_length, n))` to expand the scalar to a 2D matrix.

6. **Apply band part**: Use `F.band_part(mask, num_lower=None, num_upper=0, exclude=True)` to create the lower triangular pattern. This keeps 0s on and below the diagonal, `-inf` above.

**Implementation** (`step_03.py`):

```python
{{#include ../../steps/step_03.py}}
```

### Validation

Run `pixi run s03` to verify your implementation.

<details>
<summary>Show solution</summary>

```python
{{#include ../../solutions/solution_03.py}}
```

</details>

**Next**: In [Step 04](./step_04.md), you’ll implement multi-head attention.
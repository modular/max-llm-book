# Multi-head attention

<div class="note">

Implement scaled dot-product
[attention](https://docs.modular.com/glossary/ai/attention/) with multiple
heads, enabling the model to attend to different representation subspaces.

</div>

`GPT2MultiHeadAttention` runs 12 attention operations in parallel. Instead of
computing attention once over the full 768-dimensional space, it splits the
dimensions into 12 heads of 64 dimensions each. Each head independently learns
to focus on different patterns—syntactic structure, semantic similarity,
positional relationships, and so on.

GPT-2 uses 12 heads with 768-dimensional embeddings, giving each head 768 ÷ 12
= 64 dimensions. The Q, K, V tensors are reshaped to split the embedding across
heads, attention is computed for all heads in parallel via broadcasting, then
the outputs are concatenated back. The whole computation happens in a single
efficient sequence of tensor operations.

## Head splitting and merging

**Splitting** transforms from `[batch, seq_length, 768]` to
`[batch, 12, seq_length, 64]`. First reshape to add the head dimension:
`[batch, seq_length, 12, 64]`, then transpose to move heads before the sequence
dimension: `[batch, 12, seq_length, 64]`. Now each of the 12 heads operates
independently on its 64-dimensional subspace.

**Merging** reverses the process: transpose back to
`[batch, seq_length, 12, 64]`, then reshape to flatten the head dimension:
`[batch, seq_length, 768]`. This concatenates all head outputs back into the
original dimension.

## Scaled dot-product attention

With shape `[batch, num_heads, seq_length, head_dim]`, computing attention for
all heads simultaneously is just a matrix multiplication across the last two
dimensions. The scaling factor `1 / sqrt(head_dim)` prevents the dot products
from growing too large as head dimension increases, which would push softmax
into regions with very small gradients.

The causal mask from Section 3 is added to the attention scores before softmax,
masking out future positions.

After the output projection (`c_proj`), the model can mix information across
heads—combining the different perspectives each head learned.

The layer names `c_attn` (combined Q/K/V projection) and `c_proj` (output
projection) match Hugging Face's GPT-2 implementation for weight loading.

## The code

```python
{{#include ../../gpt2.py:multi_head_attention}}
```

`F.split` divides the combined Q/K/V projection into three equal tensors along
the last axis. The `cast` calls are needed because MAX's type system requires
explicit casts at certain boundaries between `Tensor` and `TensorValue`.

**Next**: [Section 5](./step_05.md) implements layer normalization, which
normalizes activations before each sublayer in the transformer block.

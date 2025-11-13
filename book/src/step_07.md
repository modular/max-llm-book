# Step 07: Multi-head attention

<div class="note">

Learn to use multi-head
[attention](https://docs.modular.com/glossary/ai/attention/), enabling the model
to attend to different representation subspaces.

</div>

## Building multi-head attention

In this step, you'll implement the `GPT2MultiHeadAttention` class that runs 12
attention operations in parallel. Instead of computing attention once over the
full 768-dimensional space, you split the dimensions into 12 heads of 64
dimensions each. Each head learns to focus on different patterns.

GPT-2 uses 12 heads with 768-dimensional embeddings, giving each head 768 ÷ 12 =
64 dimensions. The Q, K, V tensors are reshaped to split the embedding dimension
across heads, attention is computed for all heads in parallel, then the outputs
are concatenated back together. This happens in a single efficient operation
using tensor reshaping and broadcasting.

Multiple heads let the model learn complementary attention strategies. Different
heads can specialize in different relationships, such as one that might attend
to adjacent tokens, another to syntactic patterns, and another to semantic
similarity. This increases the model's capacity without dramatically increasing
computation.

## Understanding the architecture

Multi-head attention splits the embedding dimension, computes attention
independently for each head, then merges the results. This requires careful
tensor reshaping to organize the computation efficiently.

**Head splitting**: Transform from `[batch, seq_length, 768]` to
`[batch, 12, seq_length, 64]`. First reshape to add the head dimension:
`[batch, seq_length, 12, 64]`. Then transpose to move heads before sequence:
`[batch, 12, seq_length, 64]`. Now each of the 12 heads operates independently
on its 64-dimensional subspace.

**Parallel attention**: With shape `[batch, num_heads, seq_length, head_dim]`,
you can compute attention for all heads simultaneously. The matrix
multiplication `Q @ K^T` operates on the last two dimensions
`[seq_length, head_dim] @ [head_dim, seq_length]`, broadcasting across the batch
and head dimensions. All 12 heads computed in a single efficient operation.

**Head merging**: Reverse the splitting to go from
`[batch, 12, seq_length, 64]` back to `[batch, seq_length, 768]`. First
transpose to `[batch, seq_length, 12, 64]`, then reshape to flatten the head
dimension: `[batch, seq_length, 768]`. This concatenates all head outputs back
into the original dimension.

**Output projection (`c_proj`)**: After merging heads, apply a learned linear
transformation that maps `[batch, seq_leßngth, 768]` to
`[batch, seq_length, 768]`. This lets the model mix information across heads,
combining the different perspectives each head learned.

The layer names `c_attn` (combined Q/K/V projection) and `c_proj` (output
projection) match Hugging Face's GPT-2 implementation. This naming is essential
for loading pretrained weights.

<div class="note">

<div class="title">MAX operations</div>

You'll use the following MAX operations to complete this task:

**Linear layers**:
- [`Linear(in_features, out_features, bias=True)`](https://docs.modular.com/max/api/python/nn/module_v3#max.nn.module_v3.Linear): Q/K/V and output projections

**Tensor operations**:
- `tensor.reshape(new_shape)`: Splits or merges head dimension
- `tensor.transpose(axis1, axis2)`: Rearranges dimensions for parallel attention
- [`F.split(tensor, split_sizes, axis)`](https://docs.modular.com/max/api/python/experimental/functional#max.experimental.functional.split): Divides Q/K/V from combined projection

</div>

## Implementing multi-head attention

You'll create the `GPT2MultiHeadAttention` class with helper methods for splitting and merging heads. The implementation builds on the attention mechanism from Step 02, extending it to work with multiple heads in parallel.

First, import the required modules. You'll need `math` for scaling, `functional as F` for operations, `Tensor` for type hints, device and dtype utilities, and `Linear` and `Module` from MAX's neural network module. You'll also reuse the `causal_mask` function from Step 02.

In the `__init__` method, create the projection layers and store configuration:
- Combined Q/K/V projection: `Linear(embed_dim, 3 * embed_dim, bias=True)` stored as `self.c_attn`
- Output projection: `Linear(embed_dim, embed_dim, bias=True)` stored as `self.c_proj`
- Store `self.num_heads` (12) and `self.head_dim` (64) from config
- Calculate `self.split_size` for splitting Q, K, V later

Implement `_split_heads` to reshape for parallel attention:
- Calculate new shape by replacing the last dimension: `tensor.shape[:-1] + [num_heads, attn_head_size]`
- Reshape to add the head dimension: `tensor.reshape(new_shape)`
- Transpose to move heads to position 1: `tensor.transpose(-3, -2)`
- Returns shape `[batch, num_heads, seq_length, head_size]`

Implement `_merge_heads` to concatenate head outputs:
- Transpose to move heads back: `tensor.transpose(-3, -2)`
- Calculate flattened shape: `tensor.shape[:-2] + [num_heads * attn_head_size]`
- Reshape to merge heads: `tensor.reshape(new_shape)`
- Returns shape `[batch, seq_length, n_embd]`

Implement `_attn` to compute scaled dot-product attention for all heads:
- Compute attention scores: `query @ key.transpose(-2, -1)`
- Scale by square root of head dimension
- Apply causal mask to prevent attending to future positions
- Apply softmax to get attention weights
- Multiply weights by values: `attn_weights @ value`

In the `forward` method, orchestrate the complete multi-head attention:
- Project to Q/K/V: `qkv = self.c_attn(hidden_states)`
- Split into separate tensors: `F.split(qkv, [self.split_size, self.split_size, self.split_size], axis=-1)`
- Split heads for each: `query = self._split_heads(query, self.num_heads, self.head_dim)` (repeat for key, value)
- Compute attention: `attn_output = self._attn(query, key, value)`
- Merge heads: `attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)`
- Final projection: `return self.c_proj(attn_output)`

**Implementation** (`step_07.py`):

```python
{{#include ../../steps/step_07.py}}
```

### Validation

Run `pixi run s07` to verify your implementation.

<details>
<summary>Show solution</summary>

```python
{{#include ../../solutions/solution_07.py}}
```

</details>

**Next**: In [Step 08](./step_08.md), you'll implement residual connections and
layer normalization to enable training deep transformer networks.

# Step 06: Position embeddings

<div class="note">

Learn to create position embeddings that encode the order of tokens in a sequence.

</div>

## Implementing position embeddings

In this step you'll create position embeddings to encode where each token
appears in the sequence. While token embeddings tell the model "what" each token
is, position embeddings tell it "where" the token is located. These position
vectors are added to token embeddings before entering the transformer blocks.

Transformers process all positions in parallel through attention, unlike
Recurrent Neural Networks (RNNs) that process sequentially. This parallelism
enables faster training but loses positional information. Position embeddings
restore this information so the model can distinguish "dog bites man" from "man
bites dog".

## Understanding position embeddings

Position embeddings work like token embeddings: a lookup table with shape
[1024, 768] where 1024 is the maximum sequence length. Position 0 gets the first
row, position 1 gets the second row, and so on.

GPT-2 uses learned position embeddings, meaning these vectors are initialized
randomly and trained alongside the model. This differs from the original
Transformer which used fixed sinusoidal position encodings. Learned embeddings
let the model discover optimal position representations for its specific task,
though they cannot generalize beyond the maximum length seen during training
(1024 tokens).

**Key parameters**:
- Maximum sequence length: 1,024 positions
- Embedding dimension: 768 for GPT-2 base
- Shape: [n_positions, n_embd]
- Layer name: `wpe` (word position embeddings)

<div class="note">

<div class="title">MAX operations</div>

You'll use the following MAX operations to complete this task:

**Position indices**:
- [`Tensor.arange(seq_length, dtype, device)`](https://docs.modular.com/max/api/python/experimental/tensor#max.experimental.tensor.Tensor.arange): Creates sequence positions [0, 1, 2, ..., seq_length-1]

**Embedding layer**:
- [`Embedding(num_embeddings, dim)`](https://docs.modular.com/max/api/python/nn/module_v3#max.nn.module_v3.Embedding): Same class as token embeddings, but for positions

</div>

## Implementing the class

You'll implement the position embeddings in several steps:

1. **Import required modules**: Import `Tensor`, `Embedding`, and `Module` from
   MAX libraries.

2. **Create position embedding layer**: Use
   `Embedding(config.n_positions, dim=config.n_embd)` and store in `self.wpe`.

3. **Implement forward pass**: Call `self.wpe(position_ids)` to lookup position
   embeddings. Input shape: [seq_length] or [batch, seq_length]. Output shape:
   [seq_length, n_embd] or [batch, seq_length, n_embd].

**Implementation** (`step_06.py`):

```python
{{#include ../../steps/step_06.py}}
```

### Validation

Run `pixi run s06` to verify your implementation.

<details>
<summary>Show solution</summary>

```python
{{#include ../../solutions/solution_06.py}}
```

</details>

**Next**: In [Step 07](./step_07.md), you'll implement multi-head attention.

# Step 05: Token embeddings

<div class="note">

Learn to create token embeddings that convert discrete token IDs into continuous vector representations.

</div>

## Implementing token embeddings

In this step you'll create the `Embedding` class. This converts discrete token
IDs (integers) into continuous vector representations that the model can
process. The embedding layer is a lookup table with shape [50257, 768] where
50257 is GPT-2's vocabulary size and 768 is the embedding dimension.

Neural networks operate on continuous values, not discrete symbols. Token
embeddings convert discrete token IDs into dense vectors that can be processed
by matrix operations. During training, these embeddings naturally cluster
semantically similar words closer together in vector space.

## Understanding embeddings

The embedding layer stores one vector per vocabulary token. When you pass in
token ID 1000, it returns row 1000 as the embedding vector. The layer name `wte`
stands for "word token embeddings" and matches the naming in the original GPT-2
code for weight loading compatibility.

**Key parameters**:
- Vocabulary size: 50,257 tokens (byte-pair encoding)
- Embedding dimension: 768 for GPT-2 base
- Shape: [vocab_size, embedding_dim]

<div class="note">

<div class="title">MAX operations</div>

You'll use the following MAX operations to complete this task:

**Embedding layer**:
- [`Embedding(num_embeddings, dim)`](https://docs.modular.com/max/api/python/nn/module_v3#max.nn.module_v3.Embedding): Creates embedding lookup table with automatic weight initialization

</div>

## Implementing the class

You'll implement the `Embedding` class in several steps:

1. **Import required modules**: Import `Embedding` and `Module` from MAX libraries.

2. **Create embedding layer**: Use `Embedding(config.vocab_size, dim=config.n_embd)` and store in `self.wte`.

3. **Implement forward pass**: Call `self.wte(input_ids)` to lookup embeddings. Input shape: [batch_size, seq_length]. Output shape: [batch_size, seq_length, n_embd].

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

**Next**: In [Step 06](./step_06.md), you'll implement position embeddings to encode sequence order information, which will be combined with these token embeddings.

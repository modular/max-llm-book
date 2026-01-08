# Input embeddings

<div class="note">

Learn to create token embeddings that convert discrete token IDs into continuous vector representations.

</div>

In this step you'll create the `GPT2Model` class that will ultimately define
the whole body of the GPT-2 model, and start by converting the input token IDs
into embeddings that encode both the token and its position in the sequence.

## Understanding input embeddings

The GPT-2 model accepts input as token IDs (integers) that represent each
subword from the input text. So the first step in the `GPT2Model` must convert
these tokens to embeddings: continuous vector representations that the model
can process. This includes two types of embeddings to represent the input:

- **Token embedding**: This is the basic vector representation of a
  token (a word or subword), which is learned during training to encode a
  wide variety of meaning about that token.

  Key parameters:

  - Vocabulary size: 50,257 tokens (byte-pair encoding)
  - Embedding dimension: 768 for GPT-2 base
  - Shape: [vocab_size, embedding_dim]

- **Position embedding**: This is an additional vector sequence that's added
  onto each token embedding to tell the model the position of each token
  in the sequence.

  Key parameters:

  - Maximum sequence length: 1,024 positions
  - Embedding dimension: 768 for GPT-2 base
  - Shape: [n_positions, n_embd]
  - Layer name: `wpe` (word position embeddings)

While token embeddings tell the model "what" each token
is, position embeddings tell it "where" the token is located. These position
vectors are added to token embeddings before entering the transformer blocks.

SCOTT STOPPED HERE JAN 7

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

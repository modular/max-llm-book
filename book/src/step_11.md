# Step 11: Language model head

<div class="note">

Learn to add the final linear projection layer that converts hidden states to
vocabulary logits for next-token prediction.

</div>

## Adding the language model head

In this step, you'll create the `MaxGPT2LMHeadModel` - the complete language
model that can predict next tokens. This class wraps the transformer from Step
10 and adds a final linear layer that projects 768-dimensional hidden states to
50,257-dimensional vocabulary logits.

The language model head is a single linear layer without bias. For each position
in the sequence, it outputs a score for every possible next token. Higher scores
indicate the model thinks that token is more likely to come next.

At 768 × 50,257 = 38.6M parameters, the LM head is the single largest component
in GPT-2, representing about 33% of the model's 117M total parameters. This is
larger than all 12 transformer blocks combined.

## Understanding the projection

The language model head performs a simple linear projection using MAX's
[`Linear`](https://docs.modular.com/max/api/python/nn/module_v3#max.nn.module_v3.Linear)
layer. It maps each 768-dimensional hidden state to 50,257 scores, one per
vocabulary token.

The layer uses `bias=False`, meaning it only has weights and no bias vector.
This saves 50,257 parameters (about 0.4% of model size). The bias provides
little benefit because the layer normalization before the LM head already
centers the activations. Adding a constant bias to all logits wouldn't change
the relative probabilities after softmax.

The output is called "logits," which are raw scores before applying softmax.
Logits can be any real number. During text generation (Step 12), you'll convert
logits to probabilities with softmax. Working with logits directly enables
techniques like temperature scaling and top-k sampling.

## Understanding the complete model

With the LM head added, you now have the complete GPT-2 architecture:

1. **Input**: Token IDs `[batch, seq_length]`
2. **Embeddings**: Token + position `[batch, seq_length, 768]`
3. **Transformer blocks**: 12 blocks process the embeddings `[batch, seq_length, 768]`
4. **Final layer norm**: Normalizes the output `[batch, seq_length, 768]`
5. **LM head**: Projects to vocabulary `[batch, seq_length, 50257]`
6. **Output**: Logits `[batch, seq_length, 50257]`

Each position gets independent logits over the vocabulary. To predict the next
token after position i, you look at the logits at position i. The highest
scoring token is the model's top prediction.

<div class="note">

<div class="title">MAX operations</div>

You'll use the following MAX operations to complete this task:

**Linear layer**:
- [`Linear(in_features, out_features, bias=False)`](https://docs.modular.com/max/api/python/nn/module_v3#max.nn.module_v3.Linear): Projects hidden states to vocabulary logits

</div>

## Implementing the language model

You'll create the `MaxGPT2LMHeadModel` class that wraps the transformer with a
language modeling head. The implementation is straightforward, with just two
components and a simple forward pass.

First, import the required modules. You'll need `Linear` and `Module` from MAX,
plus the previously implemented `GPT2Config` and `GPT2Model`.

In the `__init__` method, create two components:
- Transformer: `GPT2Model(config)` stored as `self.transformer`
- LM head: `Linear(config.n_embd, config.vocab_size, bias=False)` stored as `self.lm_head`

Note the `bias=False` parameter, which creates a linear layer without bias
terms.

In the `forward` method, implement a simple two-step process:
1. Get hidden states from the transformer: `hidden_states = self.transformer(input_ids)`
2. Project to vocabulary logits: `logits = self.lm_head(hidden_states)`
3. Return `logits`

That's it. The model takes token IDs and returns logits. In the next step, you'll use these logits to generate text.

**Implementation** (`step_11.py`):

```python
{{#include ../../steps/step_11.py}}
```

### Validation

Run `pixi run s11` to verify your implementation.

<details>
<summary>Show solution</summary>

```python
{{#include ../../solutions/solution_11.py}}
```

</details>

**Next**: In [Step 12](./step_12.md), you'll implement text generation using
sampling and temperature control to generate coherent text autoregressively.

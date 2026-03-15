# Language model head

<div class="note">

Add the final linear projection layer that converts hidden states to vocabulary
logits for next-token prediction.

</div>

`MaxGPT2LMHeadModel` wraps the transformer body with a single linear layer that
projects 768-dimensional hidden states to 50,257-dimensional vocabulary logits.
This completes the GPT-2 architecture.

## The projection

For each position in the sequence, the language model head outputs a score for
every possible next token. Higher scores mean the model thinks that token is
more likely to come next. These scores are called _logits_—raw values before
softmax, which can be any real number.

The layer uses `bias=False`, omitting the bias vector. Layer normalization
before the head already centers the activations, so a constant bias adds
nothing to the relative scores after softmax. Omitting it saves 50,257
parameters.

At 768 × 50,257 = 38.6M parameters, the LM head is the largest single
component in GPT-2—about 33% of the model's 117M total parameters, more than
all 12 transformer blocks combined.

## The complete model pipeline

With the LM head, the full data flow is:

| Stage                       | Shape                        |
|-----------------------------|------------------------------|
| Input token IDs             | `[batch, seq_length]`        |
| Token + position embeddings | `[batch, seq_length, 768]`   |
| 12 transformer blocks       | `[batch, seq_length, 768]`   |
| Final layer norm            | `[batch, seq_length, 768]`   |
| LM head                     | `[batch, seq_length, 50257]` |

Each position gets independent logits over the vocabulary. To predict the next
token after position _i_, look at the logits at position _i_. The
highest-scoring token is the model's top prediction.

## The code

```python
{{#include ../../gpt2.py:language_model_head}}
```

The `forward` method reuses the parameter name `input_ids` for the transformer
output—by the time the LM head runs, it holds hidden states rather than IDs,
but the name reflects its origin.

**Next**: [Section 9](./step_09.md) covers tokenization: converting between
text strings and the token ID sequences the model operates on.

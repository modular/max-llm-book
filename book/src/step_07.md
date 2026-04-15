# Stack transformer blocks

<div class="note">

Stack 12 transformer blocks with embeddings and final normalization to create
the complete body of the GPT-2 model.

</div>

`MaxGPT2Model` is the body of GPT-2. It converts raw token IDs to embeddings,
adds position information, passes through all 12 transformer blocks, and
normalizes the final output.

The model processes input in four stages:

1. **Token embeddings**: Convert each token ID to a 768-dimensional vector via
   a learned lookup table with 50,257 entries.
2. **Position embeddings**: Add a learned position vector for each token's
   position (0 to 1,023). These are added element-wise to the token embeddings
   so the model knows token order.
3. **Transformer blocks**: Pass through 12 identical `GPT2Block` layers
   sequentially. Each block refines the representation.
4. **Final layer norm**: Normalize the output before the language model head.

## Layer depth

GPT-2 uses 12 layers because this depth allows complex pattern learning while
remaining trainable. Early layers tend to capture surface-level patterns like
word shapes and punctuation; later layers capture higher-level semantic
patterns. The representations from all layers contribute to the final output.

## Module composition

[`Sequential`](https://docs.modular.com/max/api/python/generated/max.experimental.nn.Sequential/)
chains the 12 transformer blocks in order, passing each block's output to the
next. The `*` in
`Sequential(*(GPT2Block(config) for _ in range(config.n_layer)))` unpacks the
generator as positional arguments.

[`Tensor.arange`](https://docs.modular.com/max/api/python/generated/max.experimental.tensor.Tensor#max.experimental.tensor.Tensor.arange)
generates position indices `[0, 1, ..., seq_length-1]` matching the input's
dtype and device so they're compatible for embedding lookup.

[`Embedding(vocab_size, dim)`](https://docs.modular.com/max/api/python/generated/max.experimental.nn.Embedding/)
is used for both token and position embeddings.

## MaxGPT2Model

`MaxGPT2Model` combines token embeddings, position embeddings, 12 transformer
blocks, and final layer normalization into the complete model body:

```python:gpt2.py
{{#include ../../gpt2_arch/gpt2.py:stacking_transformer_blocks}}
```

The `_` in `_, seq_length = input_ids.shape` discards the batch dimension; only
the sequence length is needed to generate position indices. The output is a
`[batch, seq_length, 768]` tensor: one contextualized representation per token
position, ready for the language model head to project into vocabulary logits.

**Next**: [Language model head](./step_08.md) adds the final projection layer
that maps these 768-dimensional hidden states to vocabulary logits.

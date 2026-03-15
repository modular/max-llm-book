# Transformer block

<div class="note">

Combine attention, MLP, layer normalization, and residual connections into a
complete transformer block.

</div>

`GPT2Block` is the repeating unit of GPT-2. It wires together all the
components from the previous sections: layer normalization, multi-head
attention, and the feed-forward network, connected by residual connections.

GPT-2 stacks 12 identical copies of this block. Each refines the representation
produced by the previous block, building from surface-level patterns in early
layers to abstract semantic understanding in later layers.

## The pre-norm pattern

Each sublayer follows the same structure: normalize first, apply the sublayer,
then add the original input back:

```text
x = x + sublayer(layer_norm(x))
```

This is called pre-normalization. GPT-2 uses it because normalizing before each
sublayer (rather than after) gives more stable gradients in deep networks—the
residual connection provides a direct path for gradients to flow backward
through all 12 blocks without passing through the normalization.

The pattern happens twice per block:

1. **Attention**: `hidden_states = attn_output + residual` (where `residual`
   is the pre-norm input)
2. **MLP**: `hidden_states = residual + feed_forward_hidden_states`

The block maintains a constant 768-dimensional representation throughout. Input
shape `[batch, seq_length, 768]` is unchanged after each sublayer, which is
essential for stacking 12 blocks together.

## Component names

`ln_1`, `attn`, `ln_2`, and `mlp` match Hugging Face's GPT-2 implementation
exactly. This naming is required for loading pretrained weights.

## The code

```python
{{#include ../../gpt2.py:transformer_block}}
```

**Next**: [Section 7](./step_07.md) stacks 12 of these blocks with embeddings
to create the main body of the GPT-2 model.

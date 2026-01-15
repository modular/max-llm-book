# Transformer block

<div class="note">

Learn to combine attention, MLP, layer normalization, and residual connections
into a complete transformer block.

</div>

In this step, you'll build a GPT-2 transformer block in the `GPT2Block` class. 
The transformer block is the definitive feature of GPT-2 and any other transformer
model. It includes a series of self-attention layers (the multi-head attention block),
a simple feed-forward network (the MLP block), and layer normalization—all of which
you’ve already built in the previous steps.

The block processes input through two sequential operations. First, it applies
layer norm, runs multi-head attention, then adds the result back to the input
(residual connection). Second, it applies another layer norm, runs the MLP, and
adds that result back. This pattern is `x = x + sublayer(layer_norm(x))`, called
pre-normalization.

GPT-2 uses pre-norm because it stabilizes training in deep networks. By
normalizing before each sublayer instead of after, gradients flow more smoothly
through the network's 12 stacked blocks.

## Understanding the components

The transformer block consists of four components, applied in this order:

**First layer norm (`ln_1`)**: Normalizes the input before attention. Uses epsilon=1e-5 for numerical stability.

**Multi-head attention (`attn`)**: The self-attention mechanism from Step 04. Lets each position attend to all previous positions.

**Second layer norm (`ln_2`)**: Normalizes before the MLP. Same configuration as the first.

**Feed-forward network (`mlp`)**: The position-wise MLP from Step 02. Expands to 3,072 dimensions internally (4× the embedding size), then projects back to 768.

The block maintains a constant 768-dimensional representation throughout. Input
shape `[batch, seq_length, 768]` stays the same after each sublayer, which is
essential for stacking 12 blocks together.

## Understanding the flow

Each sublayer follows the pre-norm pattern:

1. Save the input as `residual`
2. Apply layer normalization to the input
3. Process through the sublayer (attention or MLP)
4. Add the original `residual` back to the output

This happens twice per block, once for attention and once for the MLP. The
residual connections let gradients flow directly through the network, preventing
vanishing gradients in deep models.

Component names (`ln_1`, `attn`, `ln_2`, `mlp`) match Hugging Face's GPT-2
implementation. This matters for loading pretrained weights in later steps.

## Implementing the block

You'll create the `GPT2Block` class by composing the components from earlier
steps. The block takes `GPT2Config` and creates four sublayers, then applies
them in sequence with residual connections.

First, import the required modules. You'll need `Module` from MAX, plus the
previously implemented components: `GPT2Config`, `GPT2MLP`,
`GPT2MultiHeadAttention`, and `LayerNorm`.

In the `__init__` method, create the four sublayers:

- `ln_1`: `LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)`
- `attn`: `GPT2MultiHeadAttention(config)`
- `ln_2`: `LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)`
- `mlp`: `GPT2MLP(4 * config.n_embd, config)`

The MLP uses `4 * config.n_embd` (3,072 dimensions) as its inner dimension, following the standard transformer ratio.

In the `forward` method, implement the two sublayer blocks:

**Attention block**:

1. Save `residual = hidden_states`
2. Normalize: `hidden_states = self.ln_1(hidden_states)`
3. Apply attention: `attn_output = self.attn(hidden_states)`
4. Add back: `hidden_states = attn_output + residual`

**MLP block**:

1. Save `residual = hidden_states`
2. Normalize: `hidden_states = self.ln_2(hidden_states)`
3. Apply MLP: `feed_forward_hidden_states = self.mlp(hidden_states)`
4. Add back: `hidden_states = residual + feed_forward_hidden_states`

Finally, return `hidden_states`.

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

**Next**: In [Step 07](./step_07.md), you'll stack 12 transformer blocks together to create the main body of the GPT-2 model.

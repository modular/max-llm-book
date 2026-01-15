# Stacking transformer blocks

<div class="note">

Learn to stack 12 transformer blocks with embeddings and final normalization to create the complete GPT-2 model.

</div>

In this step, you'll create the body of the GPT-2 model (the `MaxGPT2Model` module)
as a sequence of transformer blocks (`GPT2Block`) plus `LayerNorm`. And because
the model body receives raw token IDs during inference, you’ll also have to first
convert the token IDs into token embeddings that are suitable for processing by the
transformer blocks and the rest of the neural network.

The model processes input in four stages: convert token IDs to embeddings, add position information, pass through 12 transformer blocks sequentially, and normalize the final output. Each transformer block refines the representation, building up from surface patterns in early layers to semantic understanding in later layers.

GPT-2 uses 12 layers because this depth allows the model to learn complex patterns while remaining trainable. Fewer layers would limit the model's capacity. More layers would increase training difficulty without proportional gains in quality for a 117M parameter model.

## Understanding the components

The complete model has four main components:

**Token embeddings (`wte`)**: Maps each token ID to a 768-dimensional vector using a lookup table with 50,257 entries (one per vocabulary token).

**Position embeddings (`wpe`)**: Maps each position (0 to 1,023) to a 768-dimensional vector. These are added to token embeddings so the model knows token order.

**Transformer blocks (`h`)**: 12 identical blocks stacked using MAX's [`Sequential`](https://docs.modular.com/max/api/python/nn/module_v3#max.nn.module_v3.Sequential) module. Sequential applies blocks in order, passing each block's output to the next.

**Final layer norm (`ln_f`)**: Normalizes the output after all blocks. This stabilizes the representation before the language model head (added in Step 11) projects to vocabulary logits.

## Understanding the forward pass

The forward method processes token IDs through the model:

First, create position indices using [`Tensor.arange`](https://docs.modular.com/max/api/python/experimental/tensor#max.experimental.tensor.Tensor.arange). Generate positions [0, 1, 2, ..., seq_length-1] matching the input's dtype and device. This ensures compatibility when adding to embeddings.

Next, look up embeddings. Get token embeddings with `self.wte(input_ids)` and position embeddings with `self.wpe(position_indices)`. Add them together element-wise, as both are shape `[batch, seq_length, 768]`.

Then, pass through the transformer blocks with `self.h(x)`. The
[`Sequential`](/max/api/python/nn/module_v3#max.nn.module_v3.Sequential)
module applies all 12 transformer blocks in order, each refining the representation.

Finally, normalize the output with `self.ln_f(x)` and return the result. The output shape matches the input: `[batch, seq_length, 768]`.

<div class="note">

<div class="title">MAX operations</div>

You'll use the following MAX operations to complete this task:

**Module composition**:

- [`Sequential(*modules)`](https://docs.modular.com/max/api/python/nn/module_v3#max.nn.module_v3.Sequential): Chains transformer blocks in sequence

**Embeddings**:

- [`Embedding(num_embeddings, dim)`](https://docs.modular.com/max/api/python/nn/module_v3#max.nn.module_v3.Embedding): Token and position embeddings

**Position generation**:

- [`Tensor.arange(seq_length, dtype, device)`](https://docs.modular.com/max/api/python/experimental/tensor#max.experimental.tensor.Tensor.arange): Creates position indices

</div>

## Implementing the model

You'll create the `MaxGPT2Model` class by composing embedding layers, transformer blocks, and layer normalization. The class builds on all the components from previous steps.

First, import the required modules. You'll need `Tensor` for position indices, `Embedding`, `Module`, and `Sequential` from MAX's neural network module, plus the previously implemented `GPT2Config`, `LayerNorm`, and `GPT2Block`.

In the `__init__` method, create the four components:

- Token embeddings: `Embedding(config.vocab_size, dim=config.n_embd)` stored as `self.wte`
- Position embeddings: `Embedding(config.n_positions, dim=config.n_embd)` stored as `self.wpe`
- Transformer blocks: `Sequential(*(GPT2Block(config) for _ in range(config.n_layer)))` stored as `self.h`
- Final layer norm: `LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)` stored as `self.ln_f`

The `Sequential` module takes a generator expression that creates 12 identical `GPT2Block` instances. The `*` unpacks them as arguments to `Sequential`.

In the `forward` method, implement the four-stage processing:

1. Get the sequence length from `input_ids.shape`
2. Create position indices: `Tensor.arange(seq_length, dtype=input_ids.dtype, device=input_ids.device)`
3. Look up embeddings and add them: `x = self.wte(input_ids) + self.wpe(position_indices)`
4. Apply transformer blocks: `x = self.h(x)`
5. Apply final normalization: `x = self.ln_f(x)`
6. Return `x`

The position indices must match the input's dtype and device to ensure the tensors are compatible for addition.

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

**Next**: In [Step 08](./step_08.md), you'll add the language modeling head that projects hidden states to vocabulary logits for text generation.

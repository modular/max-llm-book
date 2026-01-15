# Text generation

<div class="note">

Learn to implement autoregressive text generation with sampling and temperature control.

</div>

In this final step, you'll implement the generation loop that produces text one token at a time. The model predicts the next token, appends it to the sequence, feeds that into the model again, and repeats until reaching the desired length.

Start with a prompt like "Hello world" (tokens `[15496, 995]`). The model predicts the next token, giving you `[15496, 995, 318]` ("Hello world is"). It predicts again, producing `[15496, 995, 318, 257]` ("Hello world is a"). This process continues, with each prediction feeding back as input for the next.

You'll implement two generation strategies: greedy decoding (always pick the highest-scoring token) and sampling (randomly choose according to probabilities). You'll also add temperature control to adjust how random or focused the generation is—a higher temperature produces more variety (more hallucinations).

## Understanding the generation loop

The generation loop is simple: run the model, extract the next token prediction, append it to the sequence, repeat. Each iteration requires a full forward pass through all 12 transformer blocks.

The model outputs logits with shape `[batch, seq_length, vocab_size]`. Since you only care about predicting the next token, extract the last position: `logits[0, -1, :]`. This gives you a vector of 50,257 scores, one per vocabulary token.

These scores are logits (unnormalized), not probabilities. To convert them to probabilities, apply softmax. Then you can either pick the highest-probability token (greedy) or sample from the distribution (random).

## Understanding temperature control

Temperature scaling adjusts how random the generation is using the formula `scaled_logits = logits / temperature`.

For GPT-2, setting the temperature to 1.0 uses the original distribution. With temperature 0.7, you sharpen the distribution, and high-probability tokens become even more likely, making generation more focused and deterministic. With temperature 1.2, you flatten the distribution, and lower-probability tokens get more chances, making generation more diverse and creative. GPT-2 temperature must be between 0 and 2.0.

Temperature is applied before softmax. Dividing by a value less than 1 makes large logits even larger (sharpening), while dividing by a value greater than 1 reduces the differences between logits (flattening).

## Understanding sampling vs greedy

Greedy decoding always picks the highest-probability token using [`F.argmax`](https://docs.modular.com/max/api/python/experimental/functional#max.experimental.functional.argmax). It's fast, deterministic, and simple, but often produces repetitive text because the model keeps choosing the safest option.

Sampling randomly selects tokens according to their probabilities. Convert logits to probabilities with `F.softmax`, transfer to CPU, convert to NumPy with `np.from_dlpack`, then sample with `np.random.choice`. You use NumPy because MAX doesn't have built-in sampling yet.

Most practical generation uses sampling with temperature control. This balances creativity with coherence, as the model can explore different possibilities while still favoring high-quality continuations.

<div class="note">

<div class="title">MAX operations</div>

You'll use the following MAX operations to complete this task:

**Probability operations**:

- [`F.softmax(logits)`](https://docs.modular.com/max/api/python/experimental/functional#max.experimental.functional.softmax): Converts logits to probabilities
- [`F.argmax(logits)`](https://docs.modular.com/max/api/python/experimental/functional#max.experimental.functional.argmax): Selects highest-probability token (greedy)

**Sequence building**:

- [`F.concat([seq, new_token], axis=1)`](https://docs.modular.com/max/api/python/experimental/functional#max.experimental.functional.concat): Appends token to sequence
- [`Tensor.constant(value, dtype, device)`](https://docs.modular.com/max/api/python/experimental/tensor#max.experimental.tensor.Tensor.constant): Creates scalar tensors

**NumPy interop**:

- `probs.to(CPU())`: Transfers tensor to CPU
- `np.from_dlpack(probs)`: Converts MAX tensor to NumPy for sampling

</div>

## Implementing text generation

You'll create two functions: `generate_next_token` that predicts a single token, and `generate` that loops to produce full sequences.

First, import the required modules. You'll need `numpy` for sampling, `CPU` from MAX's driver, `DType` for type constants, `functional as F` for operations like softmax and argmax, and `Tensor` for creating tensors.

In `generate_next_token`, implement the prediction logic:

1. Run the model to get logits: `logits = model(input_ids)`
2. Extract the last position (next token prediction): `next_token_logits = logits[0, -1, :]`
3. If using temperature, scale the logits by dividing by the temperature tensor
4. For sampling: convert to probabilities with `F.softmax`, transfer to CPU, convert to NumPy with `np.from_dlpack`, sample with `np.random.choice`, then convert back to a MAX tensor
5. For greedy: use `F.argmax` to select the highest-scoring token

The temperature must be a tensor with the same dtype and device as the logits. Create it with `Tensor.constant(temperature, dtype=..., device=...)`.

In `generate`, implement the generation loop:

1. Initialize with the input: `generated_tokens = input_ids`
2. Loop `max_new_tokens` times
3. Generate the next token: `next_token = generate_next_token(model, generated_tokens, ...)`
4. Reshape to 2D: `next_token_2d = next_token.reshape([1, -1])`
5. Concatenate to the sequence: `generated_tokens = F.concat([generated_tokens, next_token_2d], axis=1)`
6. Return the complete sequence

The reshape is necessary because `concat` requires matching dimensions, and the generated token is 0D (scalar).

**Implementation** (`step_10.py`):

```python
{{#include ../../steps/step_10.py}}
```

### Validation

Run `pixi run s10` to verify your implementation.

<details>
<summary>Show solution</summary>

```python
{{#include ../../solutions/solution_10.py}}
```

</details>

**Next**: In [Step 11](./step_11.md), you'll load pretrained weights and interact with your complete GPT-2 implementation!

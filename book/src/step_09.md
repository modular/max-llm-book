# Encode and decode tokens

<div class="note">

Learn to convert between text and token IDs using tokenizers and MAX tensors.

</div>

In this step, you'll implement utility functions to bridge the gap between text and the token IDs your model operates on. The `encode_text()` function converts an input string into a tensor of token IDs, while `decode_tokens()` converts token IDs into a string.

As you saw when building the model body in step 7 (`MaxGPT2Model`), the model must receive input as token IDs (not raw text). The token IDs are integers that represent pieces of text according to a tokenizer vocabulary. GPT-2 uses a Byte Pair Encoding (BPE) tokenizer, which breaks text into subword units. For example, "Hello world" becomes `[15496, 995]` - two tokens representing the words.

You'll use the Hugging Face tokenizer to handle the text-to-token conversion, then wrap it with functions that work with MAX tensors. This separation keeps tokenization (a preprocessing step) separate from model inference (tensor operations).

## Understanding tokenization

Tokenization converts text to a list of integers. The GPT-2 tokenizer uses a vocabulary of 50,257 tokens, where common words get single tokens and rare words split into subwords.

The HuggingFace tokenizer provides an `encode` method that takes text and returns a Python list of token IDs. For example:

```python
token_ids = tokenizer.encode("Hello world")  # Returns [15496, 995]
```

You can specify `max_length` and `truncation=True` to limit sequence length. If the text exceeds `max_length`, the tokenizer cuts it off. This prevents memory issues with very long inputs.

After encoding, you need to convert the Python list to a MAX tensor. Use `Tensor.constant` to create a tensor with the token IDs, specifying `dtype=DType.int64` (GPT-2 expects 64-bit integers) and the target device.

The tensor needs shape `[batch, seq_length]` for model input. Wrap the token list in another list to add the batch dimension: `[token_ids]` becomes `[[15496, 995]]` with shape `[1, 2]`.

## Understanding decoding

Decoding reverses tokenization: convert token IDs back to text. This requires moving tensors from GPU to CPU, converting to NumPy, then using the tokenizer's `decode` method.

First, transfer the tensor to CPU with `.to(CPU())`. MAX tensors can live on GPU or CPU, but Python libraries like NumPy only work with CPU data.

Next, convert to NumPy using `np.from_dlpack`. DLPack is a standard that enables zero-copy tensor sharing between frameworks. The MAX tensor and NumPy array share the same underlying memory.

If the tensor is 2D (batch dimension present), flatten it to 1D with `.flatten()`. The tokenizer expects a flat list of token IDs, not a batched format.

Finally, convert to a Python list with `.tolist()` and decode with `tokenizer.decode(token_ids, skip_special_tokens=True)`. The `skip_special_tokens=True` parameter removes padding and end-of-sequence markers from the output.

<div class="note">

<div class="title">MAX operations</div>

You'll use the following MAX operations to complete this task:

**Tensor creation**:

- [`Tensor.constant(data, dtype, device)`](https://docs.modular.com/max/api/python/experimental/tensor#max.experimental.tensor.Tensor.constant): Creates tensor from Python data

**Device transfer**:

- [`tensor.to(CPU())`](https://docs.modular.com/max/api/python/experimental/tensor#max.experimental.tensor.Tensor.to): Moves tensor to CPU for NumPy conversion

**NumPy interop**:

- `np.from_dlpack(tensor)`: Converts MAX tensor to NumPy using DLPack protocol

</div>

## Implementing tokenization

You'll create two functions: `encode_text` to convert strings to tensors, and `decode_tokens` to convert tensors back to strings.

First, import the required modules. You'll need `numpy as np` for array operations, `CPU` from MAX's driver for device specification, `DType` for specifying integer types, and `Tensor` for creating and manipulating tensors.

In `encode_text`, implement the encoding and conversion:

1. Encode the text to token IDs using the tokenizer: `token_ids = tokenizer.encode(text, max_length=max_length, truncation=True)`
2. Convert to a MAX tensor with batch dimension: `Tensor.constant([token_ids], dtype=DType.int64, device=device)`

Note the `[token_ids]` wrapping to create the batch dimension. This gives shape `[1, seq_length]` instead of just `[seq_length]`.

In `decode_tokens`, implement the reverse process:

1. Transfer to CPU and convert to NumPy: `token_ids = np.from_dlpack(token_ids.to(CPU()))`
2. Flatten if needed: `if token_ids.ndim > 1: token_ids = token_ids.flatten()`
3. Convert to Python list: `token_ids = token_ids.tolist()`
4. Decode to text: `return tokenizer.decode(token_ids, skip_special_tokens=True)`

The flattening step handles both 1D and 2D tensors, making the function work with single sequences or batches.

**Implementation** (`step_09.py`):

```python
{{#include ../../steps/step_09.py}}
```

### Validation

Run `pixi run s09` to verify your implementation correctly converts text to tensors and back.

<details>
<summary>Show solution</summary>

```python
{{#include ../../solutions/solution_09.py}}
```

</details>

**Next**: In [Step 10](./step_10.md), you'll implement the text generation loop that uses these functions to produce coherent text autoregressively.

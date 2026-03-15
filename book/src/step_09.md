# Encode and decode tokens

<div class="note">

Convert between text and token IDs using the HuggingFace GPT-2 tokenizer.

</div>

The model operates on integer token IDs, not raw text. `encode_text` and
`decode_tokens` bridge the gap, wrapping the HuggingFace tokenizer in a
minimal interface.

## How GPT-2 tokenizes text

GPT-2 uses Byte Pair Encoding (BPE): it breaks text into subword units drawn
from a vocabulary of 50,257 tokens. Common words get a single token; rarer
words are split into pieces. For example, "Hello world" becomes `[15496, 995]`.

The tokenizer handles all the vocabulary details. The encode and decode
functions just call it and pass the results along.

## The code

```python
{{#include ../../gpt2.py:encode_and_decode}}
```

`encode_text` returns a plain Python `list[int]`—the token IDs are kept as
Python data at this stage and only converted to a MAX tensor when needed in the
generation loop (Section 10).

`decode_tokens` takes a `list[int]` and returns a string.
`skip_special_tokens=True` removes the EOS and padding markers that GPT-2 uses
internally from the decoded text.

The functions accept the tokenizer as a parameter rather than capturing it as a
global, making them straightforward to test and reuse.

**Next**: [Section 10](./step_10.md) builds the generation loop that uses these
functions to produce text autoregressively.

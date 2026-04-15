# Pipeline model

<div class="note">

How `model.py` loads the compiled GPT-2 graph, runs it on each decode step,
and manages the growing token sequence.

</div>

To extend a pipeline and connect it to a serving layer, you'll need to subclass
[`PipelineModelWithKVCache`](https://docs.modular.com/max/api/python/generated/max.pipelines.lib.PipelineModelWithKVCache/)
and tell it how to load your model, run each decode step, and manage the token
sequence.

## Load the model

Every
[`Linear`](https://docs.modular.com/max/api/python/generated/max.experimental.nn.Linear/)
and
[`Embedding`](https://docs.modular.com/max/api/python/generated/max.experimental.nn.Embedding/)
layer in `MaxGPT2LMHeadModel` allocates tensors when constructed. Without the
lazy context, those allocations fill with random values and are immediately
discarded when the checkpoint loads.
[`F.lazy()`](https://docs.modular.com/max/api/python/generated/max.experimental.functional#max.experimental.functional.lazy)
defers all allocation inside the block: layers are declared, but nothing is
allocated until `compile()` runs.

[`default_device()`](https://docs.modular.com/max/api/python/generated/max.experimental.tensor#max.experimental.tensor.default_device)
and
[`default_dtype()`](https://docs.modular.com/max/api/python/generated/max.experimental.tensor#max.experimental.tensor.default_dtype)
set context variables that module construction code reads inside the lazy block,
so layers pick up the right device and numeric type without being passed them
explicitly.

[`compile()`](https://docs.modular.com/max/api/python/generated/max.experimental.nn.Module#max.experimental.nn.Module.compile)
runs outside the lazy block. Loading safetensors buffers inside
[`F.lazy()`](https://docs.modular.com/max/api/python/generated/max.experimental.functional#max.experimental.functional.lazy)
triggers the same memory alignment error that `_to_numpy()` in
`weight_adapters.py` solves by copying into a fresh array. Passing
`weights=state_dict` to `compile()` loads and compiles in one step after the
lazy context closes:

```python:model.py
{{#include ../../gpt2_arch/model.py:load_model}}
```

## Execute a step

`execute()` receives a `GPT2Inputs`, a dataclass with one field: `tokens`,
a `[1, seq_len]` int64 `Buffer` containing all token IDs for the current
sequence.

[`Tensor.from_dlpack()`](https://docs.modular.com/max/api/python/generated/max.experimental.tensor.Tensor#max.experimental.tensor.Tensor.from_dlpack)
converts the driver `Buffer` to a MAX `Tensor` without copying. The compiled
model returns `[1, seq_len, vocab_size]`: one logit vector per position. Only
the final position's logits are needed to sample the next token, so the output
is narrowed to `[1, vocab_size]` before being handed to MAX's serving
infrastructure, which handles sampling: temperature scaling, top-p filtering,
and token selection:

```python:model.py
{{#include ../../gpt2_arch/model.py:execute_method}}
```

## Manage the token sequence

On the first step (prefill), `prepare_initial_token_inputs()` reads the full
prompt from `ctx.tokens.all` and packages it as `GPT2Inputs`. On each decode
step, `prepare_next_token_inputs()` appends the newly sampled token to the
previous token array and returns the extended sequence.

Because GPT-2 has no incremental KV cache, every decode step re-processes the
full token history from position 0. Generating 30 tokens from a 10-token prompt
means the 11th decode step processes 20 tokens, the 12th processes 21, and so
on. The implementation stays simple at the cost of efficiency: compute grows
linearly with sequence length.

```python:model.py
{{#include ../../gpt2_arch/model.py:token_inputs}}
```

**Next**: [Architecture registration](./step_12.md) covers `arch.py` and
`__init__.py`, the three-line contract that plugs the whole package into
`max serve`.

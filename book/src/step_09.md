# Weight adaptation

<div class="note">

How MAX's typed parameter interface works, and the three mappings that load
GPT-2's Hugging Face checkpoint into it.

</div>

`compile(weights=state_dict)` maps a `dict[str, WeightData]` to the named
parameters in your module. Each key must match a parameter name exactly, and
each value must carry the right shape for that parameter. MAX enforces this at
compile time: a mismatched name leaves a parameter uninitialized; a mismatched
shape fails the compile.

The adapter produces the `dict[str, WeightData]` that `compile()` requires.
Three things satisfy that contract for GPT-2: key renaming to match MAX's
module hierarchy, matrix transposition to match `Linear`'s declared shape,
and an explicit copy of the tied embedding weight the checkpoint omits.

## MAX's typed parameter interface

MAX modules declare parameters by name through the class hierarchy.
`MaxGPT2LMHeadModel` contains a `MaxGPT2Model` called `transformer`, which
contains a list of `MaxGPT2Block` layers under `h`, each with a
`MaxGPT2Attention` called `attn`, and so on. When MAX loads weights, it walks
this hierarchy to construct the expected parameter names:
`transformer.h.0.attn.c_attn.weight`, `transformer.h.0.ln_1.weight`,
`lm_head.weight`.

[`WeightData.from_numpy(arr, name)`](https://docs.modular.com/max/api/python/generated/max.graph.weights.WeightData/)
binds an array to one of those names. The adapter builds the output dict by
producing one
[`WeightData`](https://docs.modular.com/max/api/python/generated/max.graph.weights.WeightData/)
per parameter, with the name MAX expects and an array in the shape MAX expects.
That's the entire contract: name and shape.

For any model you bring up in MAX, this is the same pattern: declare your
modules, identify the checkpoint's naming and layout conventions, and write an
adapter that produces `dict[str, WeightData]` with the keys and shapes your
modules declare. The adapter is the explicit boundary between what a checkpoint
provides and what MAX's typed parameter interface requires.

## Checkpoint mappings

**Key naming:** The Hugging Face checkpoint stores keys without the top-level
module name: `h.0.ln_1.weight`. MAX expects `transformer.h.0.ln_1.weight`. The
adapter prepends `transformer.` to any key that doesn't already have the prefix.

**Shape alignment:** OpenAI trained GPT-2 with a custom `Conv1D` layer that
stores weight matrices as `[in_features, out_features]`. MAX's `Linear`
declares its weight as `[out_features, in_features]`. Three layers are
affected: `c_attn` (the combined Q/K/V projection), `c_proj` (the attention
output projection), and `c_fc` (the MLP expansion). The adapter transposes
these before wrapping them in `WeightData`. All other weight matrices are
already in the right layout.

**Tied weight:** GPT-2's safetensors file doesn't include `lm_head.weight`.
The language model head shares its weight matrix with the token embedding
table, so the checkpoint omits it to save 38.6M parameters on disk. MAX's
module declares `lm_head.weight` as a distinct named parameter, so the adapter
adds it by copying `transformer.wte.weight` into a new array under the
`lm_head.weight` key.

The adapter copies each weight into a fresh NumPy array rather than wrapping
the original buffer. GPT-2's weights arrive as memory-mapped safetensors
buffers, read-only views into the file. `compile()` requires contiguous,
writable memory; `_to_numpy()` ensures that requirement is always met.

Two keys per transformer block are skipped entirely: `.attn.bias` and
`.attn.masked_bias`. These are pre-computed causal mask buffers, not trainable
parameters. The model computes its own causal mask at runtime from
`causal_mask()`.

## The adapter

`convert_safetensor_state_dict()` applies all three operations in a single
pass over the checkpoint keys:

```python:weight_adapters.py
{{#include ../../gpt2_arch/weight_adapters.py:book}}
```

The transpose condition checks two things: the key ends in `.weight`, and it
contains one of the three Conv1D layer names. Bias vectors, stored as
`[out_features]` in both conventions, don't need transposing; only the weight
matrices do.

**Next**: [KV cache configuration](./step_10.md) covers `model_config.py`,
which tells the serving layer how much cache to allocate before the first
token runs.

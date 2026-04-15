# Architecture registration

<div class="note">

How `arch.py` and `__init__.py` plug the serving package into `max serve`,
and what each field in `SupportedArchitecture` controls.

</div>

When you run `max serve --custom-architectures gpt2_arch`, `max serve` imports
the package and reads the `ARCHITECTURES` list, which adds GPT-2 to its model
registry.

## The package entry point

`__init__.py` states the contract:

```python:__init__.py
{{#include ../../gpt2_arch/__init__.py:book}}
```

## The architecture declaration

`arch.py` assembles the
[`SupportedArchitecture`](https://docs.modular.com/max/api/python/generated/max.pipelines.lib.SupportedArchitecture/)
that MAX registers. Each field tells the serving layer something it needs before
a request arrives:

```python:arch.py
{{#include ../../gpt2_arch/arch.py:book}}
```

**`name`:** must match the `"architectures"` field in Hugging Face's
`config.json` exactly. When you run `max serve --model gpt2`, MAX downloads
the model, reads `config.json`, and looks up that name in its registry. A
mismatch means the package never loads.

**`weight_adapters`:** maps each
[`WeightsFormat`](https://docs.modular.com/max/api/python/generated/max.graph.weights.WeightsFormat/)
to a conversion function. When MAX loads the safetensors checkpoint, it calls
`weight_adapters.convert_safetensor_state_dict` to produce the layout
`MaxGPT2LMHeadModel` expects.

**`tokenizer`:** is
[`TextTokenizer`](https://docs.modular.com/max/api/python/generated/max.pipelines.lib.TextTokenizer/),
which wraps the Hugging Face tokenizer for the model. Before any token is
processed, `max serve` calls it to convert the prompt to token IDs and, after
generation, decode the output IDs back to text.

**`config`:** points to `GPT2ArchConfig`, which provides the KV cache
dimensions covered in [KV cache configuration](./step_10.md).

**`required_arguments`:** is a hard constraint on the serving layer:
`enable_prefix_caching: False` prevents `max serve` from enabling prefix
caching for this model. GPT-2 passes the full token sequence on every decode
step rather than using an incremental KV cache, so prefix caching doesn't
apply.

## What you've built

You've built two complete layers of an LLM serving system and wired them
together.

The first layer is the model: everything from token embeddings through the
language model head, compiled to a MAX graph. The second layer is the serving
infrastructure: a weight adapter that maps Hugging Face checkpoints to MAX's
layout, a config class that tells the serving layer how much KV cache to
allocate, and a pipeline model that loads, compiles, and executes the graph on
demand.

Any `max.experimental.nn.Module` follows the same pattern to get from model
weights to a live endpoint:

1. Implement the model with `max.experimental.nn`
2. Adapt the weights with a
   [`WeightsFormat`](https://docs.modular.com/max/api/python/generated/max.graph.weights.WeightsFormat/)
   converter
3. Expose cache dimensions with an
   [`ArchConfigWithAttentionKVCache`](https://docs.modular.com/max/api/python/generated/max.pipelines.lib.interfaces.ArchConfigWithAttentionKVCache/)
   subclass
4. Wrap execution in a
   [`PipelineModelWithKVCache`](https://docs.modular.com/max/api/python/generated/max.pipelines.lib.PipelineModelWithKVCache/)
   subclass
5. Register the package as a
   [`SupportedArchitecture`](https://docs.modular.com/max/api/python/generated/max.pipelines.lib.SupportedArchitecture/)
   and pass `--custom-architectures` to `max serve`

Modern LLMs build on these same components with targeted refinements:

- **Grouped-query attention (GQA)**: share key-value pairs across multiple
  query heads to reduce memory, as in LLaMA.
- **Rotary position embeddings (RoPE)**: replace learned position embeddings
  with rotation-based encoding for better length generalization.
- **SwiGLU activation**: swap GELU for the gated linear unit variant used in
  LLaMA and Mistral.
- **Incremental KV cache**: cache key and value tensors across decode steps so
  each step processes only the new token instead of the full sequence.

Each builds directly on what you've read here.

[Run the model](./serve_first.md) to see it in action.

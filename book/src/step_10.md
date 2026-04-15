# KV cache configuration

<div class="note">

What the serving layer needs to know about GPT-2's attention layout before
the first token runs, and how `model_config.py` provides it.

</div>

GPT-2 doesn't use a KV cache. Every decode step re-processes the full token
sequence from position 0, so there's nothing to cache between steps. The
serving interface requires cache dimensions regardless.

[`PipelineModelWithKVCache`](https://docs.modular.com/max/api/python/generated/max.pipelines.lib.PipelineModelWithKVCache/),
the base class `GPT2PipelineModel` extends, requires an architecture config that
exposes cache dimensions regardless. MAX uses those dimensions to allocate cache
space as part of its serving infrastructure. `GPT2ArchConfig` satisfies that
interface; the cache is allocated, but GPT-2's forward pass never reads from or
writes to it.

## Why MAX requires this interface

Generating each new token requires attending to all previous tokens. Without
a cache, every decode step recomputes key and value tensors for the full token
history: a 10-token sequence becomes 11 on the next decode step, 12 on
the step after, and so on. A KV
cache breaks this growth: it stores the key and value tensors produced at each
step so subsequent steps can read prior context directly instead of recomputing
it. Each new step processes only the one new token.

`PipelineModelWithKVCache` is designed around this pattern. Before the first
token runs, the framework allocates cache storage for the entire model: one slot
per layer, per head, per position up to the maximum sequence length. To do that
it needs the cache dimensions upfront. That's exactly what
[`ArchConfigWithAttentionKVCache`](https://docs.modular.com/max/api/python/generated/max.pipelines.lib.interfaces.ArchConfigWithAttentionKVCache/)
requires your config to provide: how many layers, how many KV heads, how large
each head is, and the maximum sequence length.

For GPT-2 here, the cache is allocated but never used. The forward pass
recomputes every key and value tensor from scratch on each step, which works
for a small model with short sequences. In a production model,
re-processing the full history on every step makes generation quadratically
more expensive as context grows and limits how many requests the server can
handle concurrently.

When you bring up a model that uses an incremental KV cache, you'd keep the same
config structure and add cache reads and writes to the forward pass.
[`KVCacheInputs`](https://docs.modular.com/max/api/python/generated/max.nn.kv_cache.KVCacheInputs/)
are passed into each decode step, and the framework manages cache lifetimes
across requests. When your forward pass reads and writes that cache, each step
processes only the one new token. The four properties below are the same in both
cases; implementing the cache is what makes a model ready to serve at scale.

## Cache dimensions

**`num_layers`:** is the number of transformer blocks: 12 for GPT-2 small.
Each block produces its own key and value tensors, so the cache has 12 layers.

**`num_key_value_heads`:** is the number of key-value pairs per attention
layer. GPT-2 uses plain multi-head attention, where every query head has its own
key and value head, so this equals `n_head` (12). Models with grouped-query
attention (GQA) return a smaller number here. LLaMA 3.1 8B has 32 query heads
but only 8 KV heads; fewer KV heads means a smaller cache.

**`head_dim`:** is the feature size of each head: `n_embd // n_head` = 768 ÷
12 = 64. This is the depth of each cached key and value tensor.

**`model_max_seq_len`:** is the upper bound on token sequence length. GPT-2's
context window is 1,024 tokens (`n_positions`).

## The configuration class

`GPT2ArchConfig` extends
[`ArchConfigWithAttentionKVCache`](https://docs.modular.com/max/api/python/generated/max.pipelines.lib.interfaces.ArchConfigWithAttentionKVCache/),
which handles the cache allocation machinery. The subclass reads each dimension
from the Hugging Face config object:

```python:model_config.py
{{#include ../../gpt2_arch/model_config.py:book}}
```

**Next**: [Pipeline model](./step_11.md) covers `model.py`, which loads the
compiled model, runs it, and manages the token sequence between decode steps.

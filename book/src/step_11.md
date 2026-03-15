# Load weights and run model

<div class="note">

Load pretrained weights from HuggingFace and prepare the model for text
generation.

</div>

`main()` brings everything together: it loads OpenAI's pretrained GPT-2
weights, builds the MAX model, compiles two inference heads, initializes the
tokenizer, and starts an interactive session.

## Loading and transposing weights

HuggingFace loads the pretrained weights with
`GPT2LMHeadModel.from_pretrained("gpt2")`. The weights are then transferred to
the MAX model via `load_state_dict`.

There's one complication: HuggingFace's GPT-2 uses `Conv1D` for its linear
layers, which stores weights transposed relative to MAX's `Linear`
(`[in, out]` instead of `[out, in]`). The `transposed_state` loop
pre-transposes the affected layers (`c_attn`, `c_proj`, `c_fc`) before loading,
so the weights land in the correct orientation without modifying the model's
layer definitions.

## Lazy initialization

The model is constructed inside `F.lazy()`:

```python
with F.lazy():
    max_model = MaxGPT2LMHeadModel(config)
    max_model.load_state_dict(transposed_state)
```

Without `F.lazy()`, the `Embedding` and `Linear` initializers would allocate
random tensors immediately, only to discard them when `load_state_dict` replaces
them. Inside the lazy context, those random initializations are deferred—they're
never allocated or compiled. `load_state_dict` installs the real HuggingFace
weights directly, saving both time and memory.

## Compiling two heads

The model is wrapped in `GPT2SamplingHead` and `GPT2GreedyHead` (from Section
10), then each is compiled with `TensorType` inputs using symbolic dimensions:

```python
token_type = TensorType(DType.int64, ("batch", "seqlen"), device=device)
temp_type  = TensorType(dtype, [], device=device)

compiled_sampler = sampling_head.compile(token_type, temp_type)
compiled_greedy  = greedy_head.compile(token_type)
```

Symbolic dimensions (`"batch"`, `"seqlen"`) let the compiled model accept any
sequence length without recompilation. Compilation takes a few seconds but only
happens once per session.

## The full main function

```python
{{#include ../../gpt2.py:load_weights_and_run_model}}
```

With `--prompt`, the model generates 20 tokens and exits. With `--chat`, it
opens the rich terminal chat interface. Without flags, it starts an interactive
prompt loop.

## Running the model

```bash
pixi run gpt2
pixi run gpt2 -- --prompt "Once upon a time"
pixi run gpt2 -- --chat
pixi run gpt2 -- --benchmark
```

**Next**: [Section 12](./step_12.md) walks through the streaming chat
implementation—stop sequences, BPE boundary handling, and the `rich` live
rendering that makes the `--chat` mode work.

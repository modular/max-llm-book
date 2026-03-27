# Text generation

<div class="note">

Generate text autoregressively using compiled sampling and greedy decoding
heads with temperature control.

</div>

Text generation is autoregressive: the model predicts one token at a time,
appends it to the sequence, and feeds the extended sequence back in for the
next prediction.

Start with `"The quick brown fox"` (a few tokens). The model predicts the next
token, giving you one more word. It predicts again with that extended context.
This continues until you've generated the desired number of tokens.

## Compiled sampling heads

Before the generation loop, the implementation wraps the model in two thin
heads—`GPT2SamplingHead` and `GPT2GreedyHead`—and compiles each one. The
compiled callables are what the generation loop actually calls:

```python
{{#include ../../gpt2.py:sampling_heads}}
```

`GPT2SamplingHead.forward` takes `input_ids` and a `temperature` tensor.
It runs the full model, extracts the last position's logits, divides by
temperature, and returns log-probabilities as `float32`—all inside the compiled
graph, with no eager MAX ops outside the graph boundary.

`GPT2GreedyHead.forward` is simpler: it runs the model and returns
`F.argmax` of the last-position logits as a scalar token ID.

Compiling these heads (done in Section 11's `main`) lets MAX optimize the full
forward pass—embedding lookups, 12 transformer blocks, layer norm, and the
projection—into a single efficient execution plan.

## Gumbel-max sampling

For stochastic generation, the implementation uses Gumbel-max sampling rather
than calling `np.random.choice` on a probability distribution. The two
approaches are mathematically equivalent, but Gumbel-max is faster: add
independent Gumbel noise to log-probabilities, then take the argmax.

One GPU→CPU transfer (via DLPack, zero-copy) and a few NumPy operations on
50,257 floats takes about 3 μs—negligible compared to the model forward pass.

## The generation loop

```python
{{#include ../../gpt2.py:text_generation}}
```

Each step:

1. Build a `[1, seq_len]` int64 tensor from the current token list using
   `np.from_dlpack` (zero-copy from numpy).
2. If sampling: call the compiled sampler, apply Gumbel noise in numpy, take
   argmax.
3. If greedy: call the compiled greedy head directly.
4. Append the new token ID to the Python list and repeat.

`rng_state` is incremented each step so consecutive tokens use different random
seeds while still being reproducible from the initial `seed`.

## Temperature

Temperature scales the log-probabilities before sampling:
`log_probs / temperature`.

- **Lower temperature** (e.g. 0.5): sharpens the distribution—the model
  strongly favors its top predictions, producing more focused text.
- **Higher temperature** (e.g. 1.2): flattens the distribution—lower-ranked
  tokens get more chances, producing more varied or surprising text.
- **Temperature = 1.0**: uses the model's unmodified distribution.

**Next**: [Section 11](./step_11.md) loads the pretrained weights and wires
everything together into a runnable model.

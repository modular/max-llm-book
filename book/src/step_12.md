# Streaming chat

<div class="note">

Build a streaming multi-turn chat interface using GPT-2 as a completion model.

</div>

GPT-2 is a _completion_ model, not an instruction-following model. It doesn't
know how to answer questions—it continues text statistically. But you can
coax it into behaving like a chat model by formatting the conversation as a
structured completion prompt and stopping generation when it starts a new
speaker turn.

The `chat` section of `gpt2.py` implements this pattern with streaming output:
each token is yielded to the terminal as it's generated, rather than waiting for
the full response.

## Prompt engineering for completion-as-chat

`_build_prompt` formats the conversation history as plain text that GPT-2 can
continue:

```python
{{#include ../../gpt2.py:chat}}
```

A two-turn conversation becomes:

```text
Human: What is the capital of France?
AI: Paris is the capital and most populous city of France.
Human: Tell me more.
AI:
```

GPT-2 completes the `AI:` line. It doesn't understand the question—it
recognizes the `Human: / AI:` pattern from training data (internet discussions,
Q&A forums) and continues it statistically. The result is often plausible but
can be confidently wrong.

History is kept as `(human, ai)` tuples and the oldest turns are evicted when
the encoded prompt exceeds 900 tokens, staying safely under GPT-2's 1,024-token
limit.

## Stop sequences

Without stopping conditions, GPT-2 would continue past the AI turn and start
generating the next human message itself. `_stream_chat_tokens` stops at
`\nHuman:` or `\nAI:`, the two markers that signal a new speaker turn.

The tricky part is partial matches. If the generated text ends with `\nH`, that
might be the start of `\nHuman:` or it might just be a newline followed by the
letter H. `_stop_prefix_len` detects this ambiguity: it returns the length of
the longest tail of the current text that is a prefix of any stop sequence.
Characters in this "hold zone" are withheld until the next token resolves
whether a full stop sequence has arrived.

## BPE boundary artifacts

The streaming loop decodes the _full generated suffix_ every step, not just the
new token:

```python
new_text = tokenizer.decode(token_ids[start_len:], skip_special_tokens=True)
```

Decoding a single GPT-2 token in isolation can produce replacement characters
(`\ufffd`) for multi-byte UTF-8 sequences. A single token may represent part of
an accented character, emoji, or other non-ASCII text that only decodes cleanly
when adjacent tokens are present. Decoding the full suffix avoids these
artifacts. The incremental delta is computed by diffing against `prev_text`,
which was set to the safe (stop-prefix-held) text from the previous step.

## Rich live rendering

`chat_loop` uses `rich.live.Live` to update the terminal panel in place as each
delta arrives:

```python
response_text = Text()
with Live(Panel(Text("…", style="dim"), ...), refresh_per_second=20) as live:
    for delta in _stream_chat_tokens(...):
        response_text.append(delta)
        live.update(Panel(response_text, ...))
```

`response_text` is a `rich.text.Text` object mutated in place each delta,
avoiding O(n²) string concatenation. The `Live` context redraws the panel at up
to 20 fps, giving the appearance of streaming output.

The per-turn `seed=turn` argument to `_stream_chat_tokens` means each
conversation turn uses a distinct but deterministic RNG seed. Two runs of the
same conversation will produce the same responses.

Run the chat interface with:

```bash
pixi run gpt2 -- --chat
```

## What's next?

You now understand the complete GPT-2 implementation from configuration to
streaming output. LLaMA, Mistral, and other modern models build on these same
components with incremental refinements:

- **Grouped-query attention (GQA)**: Share key-value pairs across multiple
  query heads to reduce memory, as in LLaMA 2.
- **Rotary position embeddings (RoPE)**: Replace learned position embeddings
  with rotation-based encoding for better length generalization.
- **SwiGLU activation**: Swap GELU for the gated linear unit variant used in
  LLaMA and PaLM.
- **Mixture of experts (MoE)**: Add sparse expert routing to scale model
  capacity efficiently, as in Mixtral.

Each builds directly on what you've read here. The attention mechanism becomes
grouped-query attention with a simple change to how key-value tensors are
reshaped. Position embeddings become RoPE by changing how positional information
is encoded. The feed-forward network becomes SwiGLU by adding a gating
mechanism.

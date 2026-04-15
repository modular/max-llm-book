# Custom Instructions for Claude

If you're using Claude (via claude.ai, Claude Code CLI, or other interfaces) to
help with this tutorial, you can provide these custom instructions to optimize
your experience.

## Instructions to share with Claude

```text
I'm working through the MAX LLM Book tutorial, which teaches building GPT-2
inference step-by-step using the MAX framework. Please help me with the
following context in mind:

**Tutorial Context:**
- This is an inference-only tutorial using pretrained weights from Hugging Face
- Training is not in scope
- The model components (Steps 01–08) live in `gpt2_arch/gpt2.py`; the full serving package is `gpt2_arch/`
- Tests run with `pixi run test` (pytest tests/ -v)
- The model serves via `pixi run serve` (max serve --custom-architectures gpt2_arch --model gpt2)

**MAX framework specifics:**
- `gpt2_arch/gpt2.py` uses `max.experimental.nn.Module` as the base class
- Functions use `@F.functional` for compiled graph definitions
- Tensor operations use `max.experimental.functional` and `max.experimental.tensor`
- Linear layers use `max.experimental.nn.Linear`
- `gpt2_arch/model.py` imports `MaxGPT2LMHeadModel` from `gpt2_arch/gpt2.py` and wraps it in `PipelineModelWithKVCache` for `max serve`, using `max.experimental.nn` throughout

**How to help me:**
1. When I ask about a section, reference the ANCHOR markers in `gpt2_arch/gpt2.py`
2. Explain MAX API patterns when I'm confused about syntax or usage
3. Help me understand transformer concepts through the working code
4. If something isn't working, help me debug with `pixi run test` output
5. Reference specific files and line numbers when explaining concepts

**What NOT to do:**
- Don't suggest training-related approaches (this is inference only)
- Don't reference `gpt2.py` at the repo root, `step_XX.py`, `pixi run gpt2`, or `pixi run chat`: those don't exist
- Don't introduce dependencies outside MAX, PyTorch, transformers, numpy

My current section: [Tell Claude which step (01-10) you're on]
```

## Example prompts to try

Here are some effective ways to ask Claude for help:

### Understanding concepts

- "Explain how multi-head attention works in the context of step 07"
- "Why do we use layer normalization before attention instead of after?"
- "What's the relationship between context_length and position embeddings?"

### Debugging check failures

```text
I'm reading step 04 (multi-head attention). Here's what I don't understand:
[describe your question]
```

### Understanding MAX API

- "What does @mo.graph do and why do we need it?"
- "How does session.load() map Hugging Face weights to my model?"
- "Explain the difference between max.experimental.nn.Linear and max.ops.linear"

### Code review

```text
Here's my implementation for step 05:
[paste your code]

Can you review it and suggest improvements?
```

### Exploring alternatives

- "Why does GPT-2 use absolute position embeddings instead of RoPE?"
- "What are the tradeoffs of different attention implementations?"
- "How would this implementation differ for GPT-3 or Llama?"

## Tips for best results

1. **Be specific about your step**: Always mention which step (01-10) you're
   working on
2. **Share test output**: Paste the full `pixi run test` output when debugging
3. **Share your code**: Let Claude see your implementation when asking for help
4. **Ask follow-ups**: If an explanation isn't clear, ask for clarification or
   examples
5. **Verify understanding**: Ask Claude to explain back what you just learned

## Using Claude Code CLI

If you're using the Claude Code CLI in your terminal:

```bash
# Navigate to the tutorial directory
cd max-llm-book

# Share the custom instructions
claude chat "I'm working through the MAX LLM Book tutorial..."

# Ask for help with a specific step
claude chat "I'm on step 04 and my tests are failing. Can you explain what token embeddings do?"

# Get explanations of check output
pixi run test 2>&1 | claude chat "Here's my check output. What's failing and why?"
```

## Resources

- [MAX documentation](https://docs.modular.com/max/)
- [Tutorial Repository](https://github.com/modular/max-llm-book)
- [GPT-2 Paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (original
  Transformer paper)

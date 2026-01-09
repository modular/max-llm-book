# Custom Instructions for Claude

If you're using Claude (via claude.ai, Claude Code CLI, or other interfaces) to
help with this tutorial, you can provide these custom instructions to optimize
your experience.

## Instructions to share with Claude

```
I'm working through the MAX LLM Book tutorial, which teaches building GPT-2
inference step-by-step using the MAX Framework. Please help me with the
following context in mind:

**Tutorial Context:**
- This is an inference-only tutorial using pretrained weights from Hugging Face
- Training is not in scope
- Each step has automated checks (pixi run s01, s02, etc.) that verify correctness
- The code is in max_llm_book/step_XX.py files
- Checks are in checks/check_step_XX.py files

**MAX Framework specifics:**
- Functions use @F.functional decorator for graph definitions
- Modules extend max.nn.module_v3.Module base class
- Tensor operations use max.experimental.functional and max.experimental.tensor
- Models use max.nn.module_v3.Linear for linear layers
- The tutorial uses MAX Framework's Python API throughout

**How to help me:**
1. When I share check failures, explain what's missing and guide me to implement it
2. Explain MAX API patterns when I'm confused about syntax or usage
3. Help me understand transformer concepts (attention, embeddings, layer norm, etc.)
4. If I ask you to write code, follow the patterns established in earlier steps
5. Reference specific files and line numbers when explaining concepts
6. If something isn't working as expected, help me debug systematically

**What NOT to do:**
- Don't suggest training-related approaches (this is inference only)
- Don't over-engineer solutions beyond what the checks require
- Don't introduce dependencies outside of MAX Framework and standard libraries
- Don't assume I need to modify check files (implementations go in step_XX.py)

My current step: [Tell Claude which step you're on]
```

## Example prompts to try

Here are some effective ways to ask Claude for help:

### Understanding concepts
- "Explain how multi-head attention works in the context of step 07"
- "Why do we use layer normalization before attention instead of after?"
- "What's the relationship between context_length and position embeddings?"

### Debugging check failures
```
I'm on step 03 and getting this check output:
[paste check output]

What do I need to implement?
```

### Understanding MAX API
- "What does @mo.graph do and why do we need it?"
- "How does session.load() map Hugging Face weights to my model?"
- "Explain the difference between max.nn.Linear and max.ops.linear"

### Code review
```
Here's my implementation for step 05:
[paste your code]

Can you review it and suggest improvements?
```

### Exploring alternatives
- "Why does this tutorial use RoPE instead of absolute position embeddings?"
- "What are the tradeoffs of different attention implementations?"
- "How would this implementation differ for GPT-3 or Llama?"

## Tips for best results

1. **Be specific about your step**: Always mention which step (01-12) you're working on
2. **Share check output**: Paste the full check output when debugging
3. **Share your code**: Let Claude see your implementation when asking for help
4. **Ask follow-ups**: If an explanation isn't clear, ask for clarification or examples
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
pixi run s05 2>&1 | claude chat "Here's my check output. What's failing and why?"
```

## Resources

- [MAX Framework Documentation](https://docs.modular.com/max/)
- [Tutorial Repository](https://github.com/modular/max-llm-book)
- [GPT-2 Paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (original Transformer paper)

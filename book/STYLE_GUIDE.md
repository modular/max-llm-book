# Tutorial Style Guide

## Philosophy

This tutorial should be direct, concise, and technically precise. We prioritize:

- **Brevity over elaboration**: Say what's needed, nothing more
- **Direct explanation**: State what you're building and why it matters, then move on
- **Minimal "why" content**: Only explain reasoning when essential to understanding
- **No fluff**: Cut conversational phrases like "Here's the interesting part" or rhetorical questions
- **Respectful tone**: Trust the reader's intelligence; no excessive validation or hand-holding

## The Three-Part Pattern

Every tutorial step follows this structure:

### 1. What You're Building (1-2 paragraphs)
**Purpose**: Directly state what you're implementing

**Guidelines**:
- Be direct: "Before you can implement X, you need to define Y"
- Inline examples rather than bullet lists when possible
- Keep it brief - 2-3 sentences max
- Use lowercase for section headers: "## Defining the model architecture"

**Example**:
```markdown
## Defining the model architecture

Before you can implement GPT-2, you need to define its architecture - the
dimensions, layer counts, and structural parameters that determine how the
model processes information.

In this step, you'll create `GPT2Config`, a class that holds all the
architectural decisions for GPT-2. This class describes things like:
embedding dimensions, number of transformer layers, and number of attention
heads. These parameters define the shape and capacity of your model.
```

### 2. Why It Matters (1 paragraph only)
**Purpose**: Explain only the most critical reason this matters

**Guidelines**:
- One paragraph maximum
- Focus on the most important practical reason
- Cut secondary explanations about reproducibility, code organization, etc.
- Be matter-of-fact

**Example**:
```markdown
OpenAI trained the original GPT-2 model with specific parameters that you
can see in the config.json file on Hugging Face. By using the exact same
values, we can later load OpenAI's pretrained weights.
```

### 3. Technical Details + Implementation
**Purpose**: Reference documentation and hands-on guide

**Guidelines**:
- Parameter lists are fine here
- Keep instruction paragraphs short
- One idea per paragraph
- Sequential, clear steps

**Example**:
```markdown
## Understanding the parameters

The GPT-2 configuration consists of seven key parameters. Each one
controls a different aspect of the model's architecture:

- `vocab_size`: Size of the token vocabulary...
- `n_positions`: Maximum sequence length...

## Implementing the configuration

Now let's implement this yourself. You'll create the `GPT2Config` class
using Python's @dataclass decorator. Dataclasses reduce boilerplate.

Instead of writing `__init__` and defining each parameter manually, you
just declare the fields with type hints and default values.
```

## Voice and Tone

### ✓ Do This

- Be direct: "Before you can implement GPT-2, you need to define its architecture"
- State facts: "OpenAI trained the original GPT-2 model with specific parameters"
- Keep paragraphs short: 2-3 sentences maximum
- Use inline lists: "things like: X, Y, and Z" instead of bullet lists
- Lowercase headers: "## Defining the model architecture"

### ✗ Avoid This

- Conversational phrases: "Here's the interesting part", "This matters for another reason"
- Rhetorical questions: "Want to experiment with a different model size later?"
- Multiple "why" paragraphs: Keep reasoning to one paragraph
- Long explanatory sections: Cut everything non-essential
- Robotic transitions: "Let's now move on to..."
- Excessive validation: "Great job!", "You did it!"
- Forced metaphors: "Think of X as Y"

## Section Headers

Use lowercase with sentence case:
- ✓ "## Defining the model architecture"
- ✓ "## Understanding the parameters"
- ✓ "## Implementing the configuration"
- ✗ "## Defining the Model Architecture"
- ✗ "## What is Model Configuration?"

## Key Differences from Typical Tutorials

This is **not** a conversational, friendly tutorial. It's **concise technical documentation** that:
- States what you're building
- Gives one reason why it matters
- Shows you how to implement it
- Moves on

No extra storytelling, no multiple "why this matters" paragraphs, no rhetorical questions.

## Quick Checklist

Before finalizing a tutorial step:

- [ ] Opens directly with what you're building (no metaphors)
- [ ] "Why it matters" section is ONE paragraph maximum
- [ ] All paragraphs are 2-3 sentences or less
- [ ] No conversational fluff ("Here's the interesting part", etc.)
- [ ] Section headers use lowercase
- [ ] Technical accuracy preserved
- [ ] Implementation steps are clear and sequential
- [ ] Total word count is minimal

## Example: Before vs After

### ❌ Before (Too Conversational)
```markdown
Here's the interesting part: these aren't arbitrary numbers. OpenAI
trained the original GPT-2 with very specific parameters, and those
parameters live in a config.json file on Hugging Face. By using the exact
same values, we can later load OpenAI's pretrained weights - meaning we
get a fully trained model without spending weeks on training infrastructure.

This matters for another reason: reproducibility. When your configuration
is separate from your implementation, you can easily save, share, and
recreate exact model architectures. Want to experiment with a different
model size later? Just swap in a different config. It also keeps your
codebase cleaner - all the hyperparameters live in one place rather than
scattered through your code.
```

### ✓ After (Direct and Concise)
```markdown
OpenAI trained the original GPT-2 model with specific parameters that you
can see in the config.json file on Hugging Face. By using the exact same
values, we can later load OpenAI's pretrained weights.
```

## Using This Guide

When writing new tutorial steps:
1. State what you're building (1-2 paragraphs)
2. Explain why it matters (1 paragraph only)
3. Provide technical details + implementation
4. Keep everything minimal and direct

See `book/src/step_01.md` for the reference implementation of this style.

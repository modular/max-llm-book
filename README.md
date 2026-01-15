# Build an LLM from scratch with MAX

Build an LLM from scratch using Modular's MAX platform.
This hands-on tutorial teaches transformer architecture through 12 progressive steps, from basic embeddings to text generation.

## What you'll learn

- **Transformer architecture**: Understand every component of GPT-2
- **MAX Python API**: Learn MAX's `nn.module_v3` for building neural networks
- **Test-driven learning**: Validate your implementation at each step
- **Production patterns**: HuggingFace-compatible architecture design

## Quick start

### Prerequisites

- [Modular MAX](https://docs.modular.com/max/) installed
- [Pixi](https://pixi.sh/) package manager
- Python 3.9+
- Basic understanding of neural networks

### Installation

```bash
# Clone or navigate to this directory
cd max-llm-book

# Install dependencies with pixi
pixi install
```

### Running the complete model

The `main.py` file contains a complete, working GPT-2 implementation that you can run:

```bash
# Run the complete pre-built model
pixi run main
```

This demonstrates how all components fit together and provides a preview of what you will build over the course of the tutorial.  When you have completed all the tutorial steps, you can run the model you built and achieve the same functionality.

```bash
# Run the complete model built over the course of the book
pixi run gpt2
```

### Running the tutorial

Each step has a skeleton file to implement and a check to verify that you have completed the step accurately:

```bash
# Run checks for a specific step
pixi run s01  # Step 1: Model configuration
pixi run s05  # Step 5: Layer normalization
pixi run s10  # Step 10: Text generation

# View the tutorial book
pixi run book
```

You can always view the book at https://llm.modular.com/ instead of running it locally.

## Tutorial structure

The tutorial follows a progressive learning path:

| Steps | Focus              | What you build                                        |
|-------|--------------------|-------------------------------------------------------|
| 00-01 | **Get started**    | Project setup and model configuration |
| 05-06 | **Build the transformer block**     | Feed-forward network, attention, layer normalization, etc  |
| 07-08 | **Assemble the model**       | Stacking transformer blocks, language model head  |
| 09-11 | **Generate text**     | Tokenization, text generation, load weights and run   |

Each step includes:

- **Conceptual explanation**: What and why
- **Implementation tasks**: Skeleton code with TODO markers
- **Validation checks**: 5-phase verification (imports, structure, implementation, placeholders, functionality)
- **Reference solution**: Complete working implementation

## Project structure

```
max-llm-book/
├── book/                  # mdBook tutorial documentation
│   └── src/
│       ├── introduction.md
│       ├── step_01.md ... step_11.md
│       └── SUMMARY.md
├── steps/                 # Skeleton files for learners
│   ├── step_01.py
│   └── ... step_11.py
├── solutions/             # Complete reference implementations
│   ├── solution_01.py
│   └── ... solution_11.py
├── checks/                # Validation checks for each step
│   ├── check_step_01.py
│   └── ... check_step_11.py
├── main.py               # Complete working GPT-2 implementation
├── pixi.toml             # Project dependencies and tasks
└── README.md             # This file
```

## How to use this tutorial

### For first-time learners

1. **Read the introduction**: `pixi run book` and read the introduction
2. **Work sequentially**: Start with Step 01 and work through in order
3. **Implement each step**: Fill in TODOs in `steps/step_XX.py`
4. **Validate with checks**: Run `pixi run sXX` to verify your implementation
5. **Compare with solution**: Check `solutions/solution_XX.py` if stuck
6. **Run your model**: After completing all steps, run `pixi run gpt2` to interact with your GPT-2!

### For experienced developers

- **Jump to specific topics**: Each step is self-contained
- **Use as reference**: Check solutions for MAX API patterns
- **Explore main.py**: See the complete implementation

## Running checks

```bash
# Check a single step
pixi run s01

# Check multiple steps
pixi run s05 && pixi run s06 && pixi run s07

# Run all checks
pixi run check-all
```

### Understanding check output

**Failed check** (skeleton code):

```
❌ Embedding is not imported from max.nn.module_v3
   Hint: Add 'from max.nn.module_v3 import Embedding, Module'
```

**Passed check** (completed implementation):

```
✅ Embedding is correctly imported from max.nn.module_v3
✅ GPT2Embeddings class exists
✅ All placeholder 'None' values have been replaced
🎉 All checks passed! Your implementation is complete.
```

## Common issues

### Import errors

```python
ModuleNotFoundError: No module named 'max'
```

**Solution**: Run `pixi install` to install MAX and dependencies.

### Check failures

If checks fail unexpectedly, ensure you're in the correct directory and have completed the step's TODOs.

### Device compatibility

The examples use CPU for simplicity.
For GPU acceleration, update `device=CPU()` to `device=GPU()` where appropriate.

## Learning resources

- **MAX Documentation**: [docs.modular.com/](https://docs.modular.com/)
- **Tutorial Book**: Run `pixi run book` for the full interactive guide
- **HuggingFace GPT-2**: [huggingface.co/gpt2](https://huggingface.co/gpt2)
- **Attention Is All You Need**: [Original transformer paper](https://arxiv.org/abs/1706.03762)

## Contributing

Found an issue or want to improve the tutorial? Contributions welcome:

1. File issues for bugs or unclear explanations
2. Suggest improvements to validation coverage
3. Add helpful examples or visualizations

## Next steps after completion

Once you've completed all the steps:

1. **Experiment with generation**: Modify temperature, sampling strategies in Step 12
2. **Analyze attention**: Visualize attention weights from your model
3. **Optimize performance**: Profile and optimize with MAX's compilation tools
4. **Build something new**: Apply these patterns to custom architectures

---

**Ready to start?** Run `pixi run book` to open the interactive tutorial, or jump straight to `pixi run s01` to begin!

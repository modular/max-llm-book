# GPT-2 in MAX: A step-by-step guide

Build GPT-2 from scratch using Modular's MAX platform. This hands-on tutorial teaches transformer architecture through 12 progressive steps, from basic embeddings to text generation.

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
cd max-gpt2

# Install dependencies with pixi
pixi install
```

### Running the tutorial

Each step has a skeleton file to implement and a test to verify:

```bash
# Run tests for a specific step
pixi run s01  # Step 1: Model configuration
pixi run s05  # Step 5: Token embeddings
pixi run s12  # Step 12: Text generation

# View the tutorial book
pixi run book
```

## Tutorial structure

The tutorial follows a progressive learning path:

| Steps | Focus | What you build |
|-------|-------|----------------|
| 01-04 | **Foundations** | Configuration, layer norm, MLP, causal masking |
| 05-06 | **Embeddings** | Token and position embeddings |
| 07 | **Attention** | Multi-head attention |
| 08-09 | **Composition** | Residual connections, transformer blocks |
| 10-12 | **Complete model** | Stacking blocks, language model head, text generation |

Each step includes:
- **Conceptual explanation**: What and why
- **Implementation tasks**: Skeleton code with TODO markers
- **Validation tests**: 5-phase verification (imports, structure, implementation, placeholders, functionality)
- **Reference solution**: Complete working implementation

## Project structure

```
max-gpt2/
├── book/                  # mdBook tutorial documentation
│   └── src/
│       ├── introduction.md
│       ├── step_01.md ... step_12.md
│       └── SUMMARY.md
├── steps/                 # Skeleton files for learners
│   ├── step_01.py
│   └── ... step_12.py
├── solutions/             # Complete reference implementations
│   ├── solution_01.py
│   └── ... solution_12.py
├── tests/                 # Validation tests for each step
│   ├── test.step_01.py
│   └── ... test.step_12.py
├── main.py               # Complete working GPT-2 implementation
├── pixi.toml             # Project dependencies and tasks
└── README.md             # This file
```

## How to use this tutorial

### For first-time learners

1. **Read the introduction**: `pixi run book` and read the introduction
2. **Work sequentially**: Start with Step 01 and work through in order
3. **Implement each step**: Fill in TODOs in `steps/step_XX.py`
4. **Validate with tests**: Run `pixi run sXX` to verify your implementation
5. **Compare with solution**: Check `solutions/solution_XX.py` if stuck

### For experienced developers

- **Jump to specific topics**: Each step is self-contained
- **Use as reference**: Check solutions for MAX API patterns
- **Explore main.py**: See the complete implementation

## Running tests

```bash
# Test a single step
pixi run s01

# Test multiple steps
pixi run s05 && pixi run s06 && pixi run s07

# Run all tests
pixi run test-all
```

### Understanding test output

**Failed test** (skeleton code):
```
❌ Embedding is not imported from max.nn.module_v3
   Hint: Add 'from max.nn.module_v3 import Embedding, Module'
```

**Passed test** (completed implementation):
```
✅ Embedding is correctly imported from max.nn.module_v3
✅ GPT2Embeddings class exists
✅ All placeholder 'None' values have been replaced
🎉 All checks passed! Your implementation is complete.
```

## Complete GPT-2 example

The `main.py` file contains a complete, working GPT-2 implementation that you can run:

```bash
# Run the complete model (requires HuggingFace weights)
pixi run huggingface
```

This demonstrates how all components fit together in production.

## Common issues

### Import errors
```python
ModuleNotFoundError: No module named 'max'
```
**Solution**: Run `pixi install` to install MAX and dependencies.

### Test failures
If tests fail unexpectedly, ensure you're in the correct directory and have completed the step's TODOs.

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
2. Suggest improvements to test coverage
3. Add helpful examples or visualizations

## Next steps after completion

Once you've completed all 12 steps:

1. **Experiment with generation**: Modify temperature, sampling strategies in Step 12
2. **Analyze attention**: Visualize attention weights from your model
3. **Optimize performance**: Profile and optimize with MAX's compilation tools
4. **Build something new**: Apply these patterns to custom architectures

---

**Ready to start?** Run `pixi run book` to open the interactive tutorial, or jump straight to `pixi run s01` to begin!

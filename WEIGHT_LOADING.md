# Loading pretrained GPT-2 weights

After completing the tutorial steps, you can load pretrained GPT-2 weights from
Hugging Face to use your MAX implementation for actual text generation.

## Quick start

The easiest way to use pretrained weights is to run the included example:

```bash
pixi run huggingface
```

This loads GPT-2 weights from HuggingFace and runs generation examples comparing MAX and PyTorch implementations.

## How weight loading works

The process involves three main steps:

### 1. Load HuggingFace weights

```python
from transformers import GPT2LMHeadModel

# Download and load pretrained weights
torch_model = GPT2LMHeadModel.from_pretrained("gpt2")
```

### 2. Transfer to MAX model

```python
from solutions.solution_01 import GPT2Config
from solutions.solution_13 import MaxGPT2LMHeadModel

# Initialize your MAX model
config = GPT2Config()
max_model = MaxGPT2LMHeadModel(config)

# Load the state dict
max_model.load_state_dict(torch_model.state_dict())
```

### 3. Transpose Conv1D weights

Hugging Face GPT-2 uses `Conv1D` layers (which store transposed weights) instead of `Linear` layers. You need to transpose these weights after loading:

```python
from max.nn.module_v3 import Linear

# Transpose weights for linear layers that correspond to Conv1D in HuggingFace
max_model.to(device)
for name, child in max_model.descendents:
    if isinstance(child, Linear):
        if any(layer_name in name for layer_name in ['c_attn', 'c_proj', 'c_fc']):
            print(f"Transposing {name}: {child.weight.shape}")
            child.weight = child.weight.T
```

This transposes weights for:

- `c_attn`: Query/Key/Value projection (Step 07, 09)
- `c_proj`: Attention output projection (Step 09)
- `c_fc`: MLP first layer (Step 04)

## Complete example

Here's a complete script to load weights and generate text:

```python
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from max.driver import CPU
from max.dtype import DType
from max.experimental.tensor import Tensor
from max.nn.module_v3 import Linear

from solutions.solution_01 import GPT2Config
from solutions.solution_13 import MaxGPT2LMHeadModel
from solutions.solution_14 import generate_tokens

# 1. Load HuggingFace model and tokenizer
torch_model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# 2. Initialize MAX model
config = GPT2Config()
max_model = MaxGPT2LMHeadModel(config)

# 3. Load and transpose weights
device = CPU()
max_model.load_state_dict(torch_model.state_dict())
max_model.to(device)

for name, child in max_model.descendents:
    if isinstance(child, Linear):
        if any(layer_name in name for layer_name in ['c_attn', 'c_proj', 'c_fc']):
            child.weight = child.weight.T

# 4. Tokenize input
prompt = "The future of artificial intelligence is"
tokens = tokenizer.encode(prompt, return_tensors="np")
input_ids = Tensor.constant(tokens, dtype=DType.int64, device=device)

# 5. Generate text
generated = generate_tokens(
    max_model,
    input_ids,
    max_new_tokens=50,
    temperature=0.8,
    do_sample=True
)

# 6. Decode output
generated_np = np.from_dlpack(generated.to(CPU()))
output_text = tokenizer.decode(generated_np[0], skip_special_tokens=True)
print(output_text)
```

## Available model sizes

You can load different GPT-2 variants by changing the model name:

| Model         | Parameters | Embedding dim | Layers | Heads |
|---------------|------------|---------------|--------|-------|
| `gpt2`        | 117M       | 768           | 12     | 12    |
| `gpt2-medium` | 345M       | 1024          | 24     | 16    |
| `gpt2-large`  | 774M       | 1280          | 36     | 20    |
| `gpt2-xl`     | 1.5B       | 1600          | 48     | 25    |

To use larger models, update the config to match:

```python
# For GPT-2 Medium
config = GPT2Config()
config.n_embd = 1024
config.n_layer = 24
config.n_head = 16
config.n_inner = 4096

torch_model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
max_model = MaxGPT2LMHeadModel(config)
# ... rest of loading process
```

## Why transpose weights?

HuggingFace's GPT-2 implementation uses `nn.Conv1D` instead of `nn.Linear` for historical reasons (matching the original OpenAI implementation). `Conv1D` in 1D is mathematically equivalent to `Linear`, but stores weights transposed:

- **Conv1D weight shape**: `[input_features, output_features]`
- **Linear weight shape**: `[output_features, input_features]`

When loading state dict, MAX interprets these as Linear weights, so we transpose them to match the expected format.

## Verification

To verify weights loaded correctly, compare outputs:

```python
# MAX output
max_logits = max_model(input_ids)

# HuggingFace output
import torch
with torch.no_grad():
    torch_logits = torch_model(torch.tensor(tokens)).logits

# Compare (should be very close)
max_np = np.from_dlpack(max_logits.to(CPU()))
torch_np = torch_logits.cpu().numpy()
print(f"Max difference: {np.abs(max_np - torch_np).max()}")
# Should be < 1e-4 (small numerical differences are expected)
```

See `main.py` for the complete working example with weight loading and generation.

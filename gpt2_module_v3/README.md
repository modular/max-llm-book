# GPT-2 Module V3 for MAX serving

This module provides a custom GPT-2 architecture for serving with `max serve` using the Module V3 API.

## Status

**Current Status: Work in Progress**

The model compiles and serves successfully, but produces incorrect (gibberish) output. The issue is still being investigated. The standalone model in `main.py` works correctly.

## How to Run

### Prerequisites

```bash
pixi install
```

### Running the Server

```bash
pixi run max serve \
  --model openai-community/gpt2 \
  --custom-architectures gpt2_module_v3 \
  --port 8888
```

> Note: We do NOT use `--use-module-v3` here because we're registering a **new** architecture. the `--use-module-v3` flag is only needed when adding a new version of an existing MAX-registered architecture (it automatically appends `_ModuleV3` to the architecture name).

### Testing the API

GPT-2 is a base language model (not a chat model), so use the completions API:

```bash
curl http://localhost:8888/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai-community/gpt2",
    "prompt": "The future of AI",
    "max_tokens": 30
  }' | jq .
```

Note: Do NOT use `/v1/chat/completions`.
GPT-2 does not have a chat template.

## Files

| File                 | Description                                              |
|----------------------|----------------------------------------------------------|
| `__init__.py`        | Exports `ARCHITECTURES` list for custom arch discovery   |
| `arch.py`            | Defines `SupportedArchitecture` for GPT-2                |
| `model.py`           | `GPT2Model` class extending `PipelineModel`              |
| `model_config.py`    | `GPT2Config` dataclass for model configuration           |
| `gpt2.py`            | Neural network module definitions (attention, MLP, etc.) |
| `weight_adapters.py` | Converts HuggingFace safetensor weights to MAX format    |

## Architecture registration

Key requirements for custom architecture registration:

1. **Export `ARCHITECTURES`**: The `__init__.py` must export an `ARCHITECTURES` list:

   ```python
   ARCHITECTURES = [gpt2]
   ```

2. **Weight Adapter**: Register a weight adapter for safetensor format:

   ```python
   weight_adapters={
       WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict,
   }
   ```

## Changes from Original Implementation

The following changes were required to adapt the model from `main.py` for serving:

### 1. Input Format Change

**Original (main.py)**: 2D input `[batch_size, seq_length]`

```python
batch_size, seq_length = input_ids.shape
```

**Serving**: 1D ragged input `[total_tokens]` with `input_row_offsets`

```python
seq_length = tokens.shape[0]  # Flattened tokens
# input_row_offsets tells where each sequence starts/ends
```

### 2. Position Embeddings

Both implementations use `Tensor.arange`:

```python
positions = Tensor.arange(seq_length, dtype=tokens.dtype, device=tokens.device)
pos_embeds = self.wpe(positions)
```

### 3. Weight Transposition

GPT-2 uses Conv1D layers which store weights as `[in_features, out_features]`, but MAX's Linear expects `[out_features, in_features]`. Required transposition for:

- `.c_attn.weight`
- `.c_proj.weight`
- `.c_fc.weight`

### 4. Weight Adapter Output Format

**Important**: The weight adapter must return **raw numpy arrays**, not `WeightData` objects:

```python
# Correct:
new_state_dict[max_name] = arr

# Wrong (causes issues):
new_state_dict[max_name] = WeightData.from_numpy(arr, max_name)
```

### 5. Contiguous Arrays

Transposed arrays must be made contiguous:

```python
arr = np.ascontiguousarray(arr.T)
```

### 6. Tied Embeddings

GPT-2 ties `lm_head.weight` to `wte.weight`:

```python
if "language_model.lm_head.weight" not in new_state_dict:
    new_state_dict["language_model.lm_head.weight"] = wte_array.copy()
```

## Developer Experience Notes

### Issues Encountered with MAX Experimental APIs

1. **`F.range` vs `Tensor.arange`**: The functional `F.range` API was deprecated/changed. Had to use `Tensor.arange` instead.

2. **DLPack Conversion**: Weight data from safetensors required careful conversion:

   ```python
   arr = np.array(np.from_dlpack(weight_data), copy=True)
   ```

3. **Non-Contiguous Tensor Errors**: MAX doesn't support non-contiguous tensors. Error message:

   ```output
   ValueError: Max does not currently support executing non-contiguous tensors.
   ```
   Solution: Always use `np.ascontiguousarray()` after transpose.

4. **Weight Adapter Return Type**: Despite type hint `dict[str, WeightData]`, the actual return must be raw data (numpy arrays), following the pattern in `gpt_oss_module_v3`.


5. **Chat Template Error**: GPT-2 is a base model without chat template. Using `/v1/chat/completions` results in:

   ```output
   ValueError: Cannot use chat template functions because tokenizer.chat_template is not set
   ```

   Solution: Use `/v1/completions` instead.

### Known Issue: Incorrect Output

The model currently produces gibberish output when served, despite:

- Weights loading correctly (verified via logging)
- Weight shapes being correct
- Transposition being applied
- Tied embeddings being handled

The standalone test comparing `main.py` model with the serving model shows identical output when weights are loaded via `load_state_dict` from PyTorch, suggesting the issue may be in how weights flow through the compile/serve pipeline.

# Load weights and run model

<div class="note">

Learn to load pretrained weights from HuggingFace and prepare the model for text generation.

</div>

With all components implemented, you're ready to load OpenAI's pretrained GPT-2 weights and run the model. This step brings everything together: loading weights from HuggingFace, handling weight format differences, initializing the tokenizer, and compiling the model for efficient inference.

The HuggingFace `transformers` library provides OpenAI's pretrained GPT-2 weights. You'll load these weights into your MAX implementation, making your model immediately capable of generating coherent text without training.

However, there's a complication: HuggingFace's GPT-2 uses Conv1D layers for its linear transformations, while your MAX implementation uses standard Linear layers. These store weights in transposed formats, so you'll need to transpose specific weight matrices after loading.

## Understanding weight loading

Weight loading involves three steps: loading the HuggingFace model, transferring weights to your MAX model, and transposing Conv1D weights.

First, load the pretrained model with `GPT2LMHeadModel.from_pretrained("gpt2")`. This downloads the weights (about 500MB) and returns a PyTorch model with the exact architecture you've implemented.

Next, transfer these weights to your MAX model using `max_model.load_state_dict(hf_model.state_dict())`. The `state_dict` is a dictionary mapping layer names to weight tensors. Since your MAX model has the exact same architecture and layer names, this transfer works seamlessly.

Finally, transpose the weights for layers that use Conv1D in HuggingFace: `c_attn`, `c_proj`, and `c_fc`. Conv1D stores weights in shape `[in_features, out_features]`, while Linear expects `[out_features, in_features]`. Use the `.T` property to transpose: `child.weight = child.weight.T`.

## Understanding model compilation

Before you can run text generation, compile the model with `.compile(token_type)`. Compilation analyzes the model's computation graph and generates optimized code for your hardware.

First, you need to specify the `token_type` input using `TensorType`. This tells the MAX compiler what shape and dtype to expect:

```python
token_type = TensorType(
    DType.int64,
    ("batch", "seqlen"),
    device=DeviceRef.from_device(device)
)
```

The shape uses symbolic dimensions `("batch", "seqlen")` rather than concrete numbers like `[1, 20]`. This allows the compiled model to handle any batch size and sequence length, not just fixed dimensions.

Compilation takes a few seconds but only happens once. After compilation, inference is much faster because MAX has optimized the entire computation graph.

## Understanding the tokenizer

Back in step 9, you implemented functions to encode and decode tokens, but both
functions require a `tokenizer` argument. Now you’ll load that tokenizer from
Hugging Face, using `GPT2Tokenizer.from_pretrained("gpt2")`,
which downloads the same tokenization rules OpenAI used during training.

Set the padding token to match the end-of-sequence token: `tokenizer.pad_token = tokenizer.eos_token`. GPT-2 doesn't have a dedicated padding token, so we reuse the EOS token for this purpose.

Then pass the `tokenizer` to the `generate_text()` function you created
in step 10 (which passes it to `tokenize_text()` and `decode_tokens()`
from step 9).

## Implementing the main function

You'll implement the `main()` function that orchestrates the entire pipeline: loading models, transferring weights, initializing the tokenizer, compiling the model, and running an interactive prompt loop.

Start by loading the pretrained HuggingFace model:

```python
hf_model = GPT2LMHeadModel.from_pretrained("gpt2")
```

Initialize your MAX model with the default device and configuration:

```python
_, device = defaults()
config = GPT2Config()
max_model = MaxGPT2LMHeadModel(config)
```

The `defaults()` function returns `(dtype, device)` tuples. You only need the device, so use `_` to ignore the dtype.

Load and transpose the weights:

```python
max_model.load_state_dict(hf_model.state_dict())
max_model.to(device)
for name, child in max_model.descendents:
    if isinstance(child, Linear):
        if any(layer_name in name for layer_name in ["c_attn", "c_proj", "c_fc"]):
            child.weight = child.weight.T
```

The `descendents` property gives you all nested modules with their full paths. Check each child's name for the Conv1D layers and transpose their weights.

Initialize the tokenizer:

```python
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
```

Compile the model:

```python
token_type = TensorType(
    DType.int64, ("batch", "seqlen"), device=DeviceRef.from_device(device)
)
compiled_max_model = max_model.compile(token_type)
```

Finally, create an interactive prompt loop where users can input text and see generated results:

```python
try:
    while True:
        user_input = input("Enter your prompt: ").strip()

        if user_input.lower() in ['quit', 'exit', 'q']:
            break

        if not user_input:
            continue

        generated_text = generate_text(
            compiled_max_model,
            tokenizer,
            device,
            user_input,
            max_new_tokens=50,
            temperature=0.8,
            do_sample=True
        )
        print(f"\nGenerated text:\n{generated_text}\n")

except KeyboardInterrupt:
    print("\n\nExiting...")
```

The loop continues until the user types 'quit', 'exit', 'q', or presses Ctrl+C.

**Implementation** (`step_11.py`):

```python
{{#include ../../steps/step_11.py}}
```

### Validation

Run `pixi run s11` to verify your implementation.

<details>
<summary>Show solution</summary>

```python
{{#include ../../solutions/solution_11.py}}
```

</details>

**Congratulations!** You've completed built a complete GPT-2 implementation from scratch. 

If code verification passed, you can execute your `step_11.py` code with `pixi run gpt2`. 

## What's next?

You now understand the architectural foundation that powers modern language
models. LLaMA, Mistral, and more build on these same components with incremental
refinements. You have everything you need to implement those refinements
yourself.

Consider extending your implementation with:

- **Grouped-query attention (GQA)**: Reduce memory consumption by sharing key-value pairs across multiple query heads, as used in LLaMA 2.
- **Rotary position embeddings (RoPE)**: Replace learned position embeddings with rotation-based encoding, improving length extrapolation in models like LLaMA and GPT-NeoX.
- **SwiGLU activation**: Swap GELU for the gated linear unit variant used in LLaMA and PaLM.
- **Mixture of experts (MoE)**: Add sparse expert routing to scale model capacity efficiently, as in Mixtral and GPT-4.

Each refinement builds directly on what you've implemented. The attention
mechanism you wrote becomes grouped-query attention with a simple modification
to how you reshape key-value tensors. Your position embeddings can be replaced
with RoPE by changing how you encode positional information. The feed-forward
network you built becomes SwiGLU by adding a gating mechanism.

Pick an architecture that interests you and start building. You'll find the
patterns are familiar because the fundamentals haven't changed.
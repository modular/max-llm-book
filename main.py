import math
from dataclasses import dataclass
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GenerationConfig

from max.driver import Device, CPU
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.tensor import Tensor, TensorType, defaults
from max.graph import Dim, DimLike, DeviceRef
from max.nn.module_v3 import (
    Embedding,
    Linear,
    Module,
    Sequential,
)

# ANCHOR: model_configuration
@dataclass
class GPT2Config:
    """GPT-2 configuration matching HuggingFace"""

    vocab_size = 50257
    n_positions = 1024
    n_embd = 768
    n_layer = 12
    n_head = 12
    n_inner = None
    layer_norm_epsilon = 1e-5

# ANCHOR_END: model_configuration

# ANCHOR: feed_forward_network
class GPT2MLP(Module):
    """Exact HuggingFace GPT-2 MLP structure"""

    def __init__(self, intermediate_size, config):
        super().__init__()
        embed_dim = config.n_embd
        self.c_fc = Linear(embed_dim, intermediate_size, bias=True)
        self.c_proj = Linear(intermediate_size, embed_dim, bias=True)

    def __call__(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = F.gelu(hidden_states, approximate="tanh")
        hidden_states = self.c_proj(hidden_states)
        return hidden_states

# ANCHOR_END: feed_forward_network

# ANCHOR: causal_mask
@F.functional
def causal_mask(
    sequence_length: DimLike,
    num_tokens: DimLike,
    *,
    dtype: DType,
    device: Device,
):
    n = Dim(sequence_length) + num_tokens
    mask = Tensor.constant(float("-inf"), dtype=dtype, device=device)
    mask = F.broadcast_to(mask, shape=(sequence_length, n))
    return F.band_part(mask, num_lower=None, num_upper=0, exclude=True)

# ANCHOR_END: causal_mask

# ANCHOR: multi_head_attention
class GPT2MultiHeadAttention(Module):
    """Exact HuggingFace GPT-2 attention structure"""

    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.n_embd
        self.num_heads = config.n_head
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim

        self.c_attn = Linear(self.embed_dim, 3 * self.embed_dim, bias=True)
        self.c_proj = Linear(self.embed_dim, self.embed_dim, bias=True)

    def _attn(self, query, key, value):
        attn_weights = query @ key.transpose(-1, -2)

        # Scale attention weights
        attn_weights = attn_weights / math.sqrt(int(value.shape[-1]))

        # Apply causal mask
        seq_len = query.shape[-2]
        mask = causal_mask(seq_len, 0, dtype=query.dtype, device=query.device)
        attn_weights = attn_weights + mask

        attn_weights = F.softmax(attn_weights)
        attn_output = attn_weights @ value

        return attn_output

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """Split the last dimension into (num_heads, head_size)"""
        new_shape = tensor.shape[:-1] + [num_heads, attn_head_size]
        tensor = tensor.reshape(new_shape)
        return tensor.transpose(-3, -2)  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """Merge attention heads back"""
        tensor = tensor.transpose(-3, -2)
        new_shape = tensor.shape[:-2] + [num_heads * attn_head_size]
        return tensor.reshape(new_shape)

    def __call__(self, hidden_states):
        query, key, value = F.split(
            self.c_attn(hidden_states),
            [self.split_size, self.split_size, self.split_size],
            axis=2,
        )

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        attn_output = self._attn(query, key, value)
        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)

        return attn_output

# ANCHOR_END: multi_head_attention

# ANCHOR: layer_normalization
class LayerNorm(Module):
    def __init__(self, dim: DimLike, *, eps: float = 1e-5):
        self.eps = eps
        self.weight = Tensor.ones([dim])
        self.bias = Tensor.zeros([dim])

    def __call__(self, x: Tensor) -> Tensor:
        return F.layer_norm(x, gamma=self.weight, beta=self.bias, epsilon=self.eps)

# ANCHOR_END: layer_normalization

# ANCHOR: transformer_block
class GPT2Block(Module):
    """Exact HuggingFace GPT-2 transformer block structure"""

    def __init__(self, config):
        super().__init__()
        hidden_size = config.n_embd
        inner_dim = (
            config.n_inner
            if hasattr(config, "n_inner") and config.n_inner is not None
            else 4 * hidden_size
        )

        self.ln_1 = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPT2MultiHeadAttention(config)
        self.ln_2 = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(inner_dim, config)

    def __call__(self, hidden_states):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output = self.attn(hidden_states)
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = residual + feed_forward_hidden_states

        return hidden_states

# ANCHOR_END: transformer_block

# ANCHOR: stacking_transformer_blocks
class MaxGPT2Model(Module):
    def __init__(
        self,
        config: GPT2Config,
    ):
        self.wte = Embedding(config.vocab_size, dim=config.n_embd)
        self.wpe = Embedding(config.n_positions, dim=config.n_embd)
        self.h = Sequential(*(GPT2Block(config) for _ in range(config.n_layer)))
        self.ln_f = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

    def __call__(self, input_ids):
        batch_size, seq_length = input_ids.shape
        tok_embeds = self.wte(input_ids)
        pos_embeds = self.wpe(
            Tensor.arange(seq_length, dtype=input_ids.dtype, device=input_ids.device)
        )
        x = tok_embeds + pos_embeds
        x = self.h(x)
        x = self.ln_f(x)
        return x

# ANCHOR_END: stacking_transformer_blocks

# ANCHOR: language_model_head
class MaxGPT2LMHeadModel(Module):
    """Exact HuggingFace GPT-2 model structure"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = MaxGPT2Model(config)
        self.lm_head = Linear(config.n_embd, config.vocab_size, bias=False)

    def __call__(self, input_ids):
        input_ids = self.transformer(input_ids)
        return self.lm_head(input_ids)

# ANCHOR_END: language_model_head

# ANCHOR: encode_and_decode
def encode_text(text: str, tokenizer, device, max_length: int = 128):
    """Tokenize text and convert to tensor."""
    token_ids = tokenizer.encode(text, max_length=max_length, truncation=True)
    return Tensor.constant([token_ids], dtype=DType.int64, device=device)

def decode_tokens(token_ids: Tensor, tokenizer):
    """Decode token IDs back to text."""
    token_ids = np.from_dlpack(token_ids.to(CPU()))
    if token_ids.ndim > 1:
        token_ids = token_ids.flatten()
    token_ids = token_ids.tolist()
    return tokenizer.decode(token_ids, skip_special_tokens=True)

# ANCHOR_END: encode_and_decode

# ANCHOR: text_generation
def generate_text(
    model,
    tokenizer,
    device,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 0.8,
    do_sample: bool = True,
):
    """Generate text using the Max model."""
    generated_tokens = encode_text(prompt, tokenizer, device, max_length=100)

    print(f"Starting generation from: '{prompt}'")
    print(
        f"Settings: max_new_tokens={max_new_tokens}, temperature={temperature}, do_sample={do_sample}"
    )
    print("-" * 50)

    for step in range(max_new_tokens):
        logits = model(generated_tokens)
        next_token_logits = logits[0, -1, :]

        if do_sample and temperature > 0:
            # Simple temperature scaling without top-k
            temp_tensor = Tensor.constant(
                temperature,
                dtype=next_token_logits.dtype,
                device=next_token_logits.device,
            )
            next_token_logits = next_token_logits / temp_tensor
            probs = F.softmax(next_token_logits)

            # Convert to numpy for actual sampling
            probs_np = np.from_dlpack(probs.to(CPU()))
            next_token_id = np.random.choice(len(probs_np), p=probs_np)
            next_token_tensor = Tensor.constant(
                next_token_id, dtype=DType.int64, device=generated_tokens.device
            )
        else:
            next_token_tensor = F.argmax(next_token_logits)

        next_token_2d = next_token_tensor.reshape([1, -1])
        generated_tokens = F.concat([generated_tokens, next_token_2d], axis=1)

        if step % 5 == 0 or step == max_new_tokens - 1:
            current_text = decode_tokens(generated_tokens, tokenizer)
            print(f"Step {step + 1:2d}: {current_text}")

    final_text = decode_tokens(generated_tokens, tokenizer)
    print("-" * 50)
    print(f"Final generated text: '{final_text}'")
    return final_text

# ANCHOR_END: text_generation

# ANCHOR: load_weights_and_run_model
def main():
    # Load HuggingFace model
    hf_model = GPT2LMHeadModel.from_pretrained("gpt2")
    print(f"Loaded HuggingFace model:\n{hf_model}")

    # Initialize Max model
    _, device = defaults()
    print(f"Using device: {device}")
    config = GPT2Config()
    max_model = MaxGPT2LMHeadModel(config)

    print(
        f"Model has {config.n_layer} layers, {config.n_head} heads, {config.n_embd} embedding dim"
    )

    # Load state dict and transpose weights
    max_model.load_state_dict(hf_model.state_dict())
    max_model.to(device)
    for name, child in max_model.descendents:
        if isinstance(child, Linear):
            if any(layer_name in name for layer_name in ["c_attn", "c_proj", "c_fc"]):
                print(f"Transposing {name}: {child.weight.shape}")
                # The upstream model has conv1d layers instead of linear, which have their weights
                # stored transposed compared to linear
                child.weight = child.weight.T

    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # Set padding token

    # Compile model
    print("\nCompiling model...")
    token_type = TensorType(
        DType.int64, ("batch", "seqlen"), device=DeviceRef.from_device(device)
    )
    compiled_max_model = max_model.compile(token_type)

    # Interactive prompt loop
    print("\n" + "=" * 50)
    print("Model ready! Enter prompts to generate text.")
    print("Press Ctrl+C or type 'quit' to exit.")
    print("=" * 50 + "\n")

    try:
        while True:
            user_input = input("Enter your prompt: ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Exiting...")
                break

            if not user_input:
                print("Please enter a non-empty prompt.\n")
                continue

            print()
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
            print("-" * 50 + "\n")

    except KeyboardInterrupt:
        print("\n\nExiting...")

# ANCHOR_END: load_weights_and_run_model

if __name__ == "__main__":
    main()

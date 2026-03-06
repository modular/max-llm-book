# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
import argparse
import math
import os
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import max.experimental.functional as F
import numpy as np
import torch
from max.driver import (
    CPU,
    Buffer,
    Device,
    accelerator_architecture_name,
    accelerator_count,
)
from max.dtype import DType
from max.experimental.nn import (
    Embedding,
    Linear,
    Module,
    Sequential,
)
from max.experimental.tensor import (
    Tensor,
    TensorType,
    defaults,
)
from max.graph import Dim, DimLike, TensorValue
from transformers import GPT2LMHeadModel, GPT2Tokenizer


# ANCHOR: model_configuration
@dataclass
class GPT2Config:
    """GPT-2 configuration matching HuggingFace"""

    vocab_size: int = 50257
    n_positions: int = 1024
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    n_inner: int | None = None
    layer_norm_epsilon: float = 1e-5


# ANCHOR_END: model_configuration


# ANCHOR: feed_forward_network
class GPT2MLP(Module):  # type: ignore[type-arg]
    """Exact HuggingFace GPT-2 MLP structure"""

    def __init__(self, intermediate_size: int, config: GPT2Config) -> None:
        embed_dim = config.n_embd
        self.c_fc = Linear(embed_dim, intermediate_size, bias=True)
        self.c_proj = Linear(intermediate_size, embed_dim, bias=True)

    def forward(self, hidden_states: Tensor) -> Tensor:
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
) -> Tensor:
    n = Dim(sequence_length) + num_tokens
    mask = Tensor(float("-inf"), dtype=dtype, device=device)
    mask = F.broadcast_to(mask, shape=(sequence_length, n))
    return F.band_part(mask, num_lower=None, num_upper=0, exclude=True)


# ANCHOR_END: causal_mask


# ANCHOR: multi_head_attention
class GPT2MultiHeadAttention(Module):  # type: ignore[type-arg]
    """Exact HuggingFace GPT-2 attention structure"""

    def __init__(self, config: GPT2Config) -> None:
        self.embed_dim = config.n_embd
        self.num_heads = config.n_head
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim

        self.c_attn = Linear(self.embed_dim, 3 * self.embed_dim, bias=True)
        self.c_proj = Linear(self.embed_dim, self.embed_dim, bias=True)

    def _attn(
        self,
        query: Tensor | TensorValue,
        key: Tensor | TensorValue,
        value: Tensor | TensorValue,
    ) -> Tensor | TensorValue:
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

    def _split_heads(
        self, tensor: Tensor | TensorValue, num_heads: int, attn_head_size: int
    ) -> Tensor | TensorValue:
        """Split the last dimension into (num_heads, head_size)"""
        new_shape = list(tensor.shape[:-1]) + [num_heads, attn_head_size]
        tensor = tensor.reshape(new_shape)
        return tensor.transpose(
            -3, -2
        )  # (batch, head, seq_length, head_features)

    def _merge_heads(
        self, tensor: Tensor | TensorValue, num_heads: int, attn_head_size: int
    ) -> Tensor | TensorValue:
        """Merge attention heads back"""
        tensor = tensor.transpose(-3, -2)
        new_shape = list(tensor.shape[:-2]) + [num_heads * attn_head_size]
        return tensor.reshape(new_shape)

    def forward(self, hidden_states: Tensor) -> Tensor:
        split_result = F.split(
            self.c_attn(hidden_states),
            [self.split_size, self.split_size, self.split_size],
            axis=2,
        )
        query = cast(Tensor | TensorValue, split_result[0])
        key = cast(Tensor | TensorValue, split_result[1])
        value = cast(Tensor | TensorValue, split_result[2])

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        attn_output = self._attn(query, key, value)
        attn_output = self._merge_heads(
            attn_output, self.num_heads, self.head_dim
        )
        attn_output = self.c_proj(cast(Tensor, attn_output))

        return cast(Tensor, attn_output)


# ANCHOR_END: multi_head_attention


# ANCHOR: layer_normalization
class LayerNorm(Module):  # type: ignore[type-arg]
    def __init__(self, dim: DimLike, *, eps: float = 1e-5) -> None:
        self.eps = eps
        self.weight = Tensor.ones([dim])
        self.bias = Tensor.zeros([dim])

    def forward(self, x: Tensor) -> Tensor:
        return F.layer_norm(
            x, gamma=self.weight, beta=self.bias, epsilon=self.eps
        )


# ANCHOR_END: layer_normalization


# ANCHOR: transformer_block
class GPT2Block(Module):  # type: ignore[type-arg]
    """Exact HuggingFace GPT-2 transformer block structure"""

    def __init__(self, config: GPT2Config) -> None:
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

    def forward(self, hidden_states: Tensor) -> Tensor:
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
class MaxGPT2Model(Module):  # type: ignore[type-arg]
    def __init__(
        self,
        config: GPT2Config,
    ) -> None:
        self.wte = Embedding(config.vocab_size, dim=config.n_embd)
        self.wpe = Embedding(config.n_positions, dim=config.n_embd)
        self.h = Sequential(*(GPT2Block(config) for _ in range(config.n_layer)))
        self.ln_f = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

    def forward(self, input_ids: Tensor) -> Tensor:
        _, seq_length = input_ids.shape
        tok_embeds = self.wte(input_ids)
        pos_embeds = self.wpe(
            Tensor.arange(
                seq_length, dtype=input_ids.dtype, device=input_ids.device
            )
        )
        x = tok_embeds + pos_embeds
        x = self.h(x)
        x = self.ln_f(x)
        return x


# ANCHOR_END: stacking_transformer_blocks


# ANCHOR: language_model_head
class MaxGPT2LMHeadModel(Module):  # type: ignore[type-arg]
    """Exact HuggingFace GPT-2 model structure"""

    def __init__(self, config: GPT2Config) -> None:
        self.config = config
        self.transformer = MaxGPT2Model(config)
        self.lm_head = Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, input_ids: Tensor) -> Tensor:
        input_ids = self.transformer(input_ids)
        return self.lm_head(input_ids)


# ANCHOR_END: language_model_head


# ANCHOR: sampling_heads
class GPT2SamplingHead(Module):  # type: ignore[type-arg]
    """Compiled forward: last-token log-probs scaled by temperature.

    Returns a float32 [vocab_size] log-probability tensor ready for Gumbel-max
    sampling. The float32 cast happens inside the compiled graph (zero overhead)
    so the caller can use numpy DLPack directly without any eager MAX ops.
    """

    def __init__(self, lm_head: MaxGPT2LMHeadModel) -> None:
        self.lm_head = lm_head  # no super().__init__() is needed for Module

    def forward(self, input_ids: Tensor, temperature: Tensor) -> Tensor:
        logits = self.lm_head(input_ids)  # [1, seq_len, vocab_size]
        last = logits[0, -1, :]  # [vocab_size]
        log_probs = F.logsoftmax(last / temperature)
        # Cast inside compiled graph — free; avoids eager cast op outside.
        return log_probs.cast(DType.float32)  # [vocab_size] float32 log-probs


class GPT2GreedyHead(Module):  # type: ignore[type-arg]
    """Compiled forward: greedy argmax, returns scalar token ID."""

    def __init__(self, lm_head: MaxGPT2LMHeadModel) -> None:
        self.lm_head = lm_head

    def forward(self, input_ids: Tensor) -> Tensor:
        logits = self.lm_head(input_ids)  # [1, seq_len, vocab_size]
        return F.argmax(logits[0, -1, :])  # scalar int64 token id


# ANCHOR_END: sampling_heads


# ANCHOR: encode_and_decode
def encode_text(
    text: str, tokenizer: GPT2Tokenizer, max_length: int = 128
) -> list[int]:
    """Tokenize text and return token IDs as a plain Python list."""
    return tokenizer.encode(text, max_length=max_length, truncation=True)


def decode_tokens(token_ids: list[int], tokenizer: GPT2Tokenizer) -> str:
    """Decode a list of token IDs back to text."""
    return tokenizer.decode(token_ids, skip_special_tokens=True)


# ANCHOR_END: encode_and_decode


# ANCHOR: text_generation
def generate_text(
    sampler: Callable[[Tensor, Tensor], Tensor],
    greedy: Callable[[Tensor], Tensor],
    tokenizer: GPT2Tokenizer,
    device: Device,
    dtype: DType,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 0.8,
    do_sample: bool = True,
    seed: int = 0,
) -> str:
    """Generate text using compiled MAX models.

    Args:
        sampler: Compiled GPT2SamplingHead — returns log-probs for stochastic
            decoding. Called as ``sampler(input_ids, temperature_tensor)``.
        greedy: Compiled GPT2GreedyHead — returns scalar token ID for greedy
            decoding. Called as ``greedy(input_ids)``.
        tokenizer: HuggingFace GPT-2 tokenizer.
        device: Target device for input tensor construction.
        dtype: Dtype for the temperature scalar.
        prompt: Text prompt to continue.
        max_new_tokens: Maximum number of new tokens to generate.
        temperature: Sampling temperature (ignored when do_sample=False).
        do_sample: If True, use Gumbel-max stochastic sampling; else greedy.
        seed: Initial RNG seed for reproducibility.

    Returns:
        The full generated string (prompt + new tokens), decoded.
    """
    token_ids: list[int] = tokenizer.encode(
        prompt, max_length=100, truncation=True
    )
    temperature_tensor = Tensor(temperature, dtype=dtype, device=device)
    rng_state = seed

    print(f"Starting generation from: '{prompt}'")
    print(
        f"Settings: max_new_tokens={max_new_tokens}, temperature={temperature},"
        f" do_sample={do_sample}"
    )
    print("-" * 50)

    for step in range(max_new_tokens):
        input_tensor = _make_token_tensor(token_ids, device)

        if do_sample:
            # Compiled: all deterministic NN ops → [vocab_size] log-probs
            log_probs = sampler(input_tensor, temperature_tensor)
            # Gumbel-max: one GPU→CPU transfer + fast numpy (~3μs for 50K floats)
            rng = np.random.default_rng(rng_state)
            token_id = _gumbel_sample(log_probs, rng)
            rng_state += 1
        else:
            token_id = int(greedy(input_tensor).item())

        token_ids.append(int(token_id))

        if step % 5 == 0 or step == max_new_tokens - 1:
            current_text = decode_tokens(token_ids, tokenizer)
            print(f"Step {step + 1:2d}: {current_text}")

    final_text = decode_tokens(token_ids, tokenizer)
    print("-" * 50)
    print(f"Final generated text: '{final_text}'")
    return final_text


# ANCHOR_END: text_generation


# ANCHOR: benchmark
_BENCH_PROMPT = "The quick brown fox jumps over"
_BENCH_N_TOKENS = 50
_BENCH_WARMUP_GPU = 3
_BENCH_WARMUP_CPU = 1


def _make_token_tensor(ids: list[int], device: Device) -> Tensor:
    """Build an int64 [1, seq_len] input tensor via driver.Buffer — no eager compilation.

    ``Tensor([ids], dtype=DType.int64, device=device)`` goes through
    ``F.constant`` (eager compiled graph) and re-compiles for every new
    sequence length. Using ``driver.Buffer.from_numpy`` + ``.to(device)``
    keeps the hot loop free of compilation overhead.
    """
    np_ids = np.array([ids], dtype=np.int64)
    cpu_buf = Buffer.from_numpy(np_ids)
    gpu_buf = cpu_buf.to(device)
    return Tensor(storage=gpu_buf)


def _gumbel_sample(log_probs: Tensor, rng: np.random.Generator) -> int:
    """Gumbel-max sampling via one GPU→CPU driver transfer + fast numpy ops.

    Uses ``driver_tensor.to(CPU())`` — a direct driver-level memory copy that
    bypasses the eager-graph compilation framework entirely. No MAX eager ops
    are created in the hot loop, so there is no JIT overhead per token.

    log_probs is float32 (cast inside the compiled graph), so numpy DLPack works
    without any additional conversion.
    """
    # driver_tensor.to(CPU()) is a direct device→host copy, not an eager compiled op.
    lp_np: np.ndarray = np.from_dlpack(log_probs.driver_tensor.to(CPU()))
    vocab_size = int(lp_np.shape[0])
    u: np.ndarray = np.asarray(
        rng.uniform(1e-20, 1.0, size=vocab_size), dtype=np.float32
    )
    gumbel: np.ndarray = -np.log(-np.log(u))
    return int(np.argmax(lp_np + gumbel))


def _one_benchmark_pass(
    compiled_sampler: Callable[[Tensor, Tensor], Tensor],
    tokenizer: GPT2Tokenizer,
    device: Device,
    dtype: DType,
    *,
    n_warmup: int,
    label: str,
) -> None:
    """Run warm-up then timed generation; print results for one interpreter mode."""
    temperature_tensor = Tensor(0.8, dtype=dtype, device=device)
    warmup_rng = np.random.default_rng(0)

    # warm-up (not timed) — primes GPU kernel dispatch, L2 cache, driver state
    for _ in range(n_warmup):
        ids = tokenizer.encode(_BENCH_PROMPT, max_length=100, truncation=True)
        for _ in range(_BENCH_N_TOKENS):
            inp = _make_token_tensor(ids, device)
            log_probs = compiled_sampler(inp, temperature_tensor)
            ids.append(_gumbel_sample(log_probs, warmup_rng))

    # timed run
    token_ids = tokenizer.encode(_BENCH_PROMPT, max_length=100, truncation=True)
    timed_rng = np.random.default_rng(9999)  # distinct seed from warmup
    step_times: list[float] = []

    for _ in range(_BENCH_N_TOKENS):
        inp = _make_token_tensor(token_ids, device)
        t0 = time.perf_counter()
        log_probs = compiled_sampler(inp, temperature_tensor)
        token_id = _gumbel_sample(log_probs, timed_rng)
        t1 = time.perf_counter()
        step_times.append((t1 - t0) * 1_000)
        token_ids.append(token_id)

    first_ms = step_times[0]
    warm_times = step_times[1:]
    mean_ms = statistics.mean(warm_times)
    stdev_ms = statistics.stdev(warm_times) if len(warm_times) > 1 else 0.0
    total_s = sum(step_times) / 1_000
    tps = _BENCH_N_TOKENS / total_s

    print(f"\n[{label}]", flush=True)
    print(
        f"  First token  : {first_ms:8.2f} ms  (may include residual JIT)",
        flush=True,
    )
    print(
        f"  Mean / token : {mean_ms:8.2f} ms  ± {stdev_ms:.2f} ms"
        f"  (steps 1-{_BENCH_N_TOKENS - 1})",
        flush=True,
    )
    print(f"  Tokens / sec : {tps:8.1f}", flush=True)
    print(
        f"  Total time   : {total_s:8.3f} s  for {_BENCH_N_TOKENS} tokens",
        flush=True,
    )


def run_benchmark(
    compiled_sampler: Callable[[Tensor, Tensor], Tensor],
    tokenizer: GPT2Tokenizer,
    device: Device,
    dtype: DType,
) -> None:
    """Run two benchmark passes: eager JIT mode and interpreter mode.

    The interpreter is controlled by the ``MAX_USE_EAGER_INTERPRETER`` env var
    (workspace/dev version). ``_default_use_interpreter()`` reads the env at
    call-time so toggling mid-process takes effect on any new eager graph.

    The compiled transformer + logsoftmax are unaffected in both modes.
    Setting the interpreter only impacts any ad-hoc eager ops built after the
    env var is set. In the current sampling path, only one GPU→CPU transfer
    (DLPack) and numpy ops execute outside the compiled graph.
    """
    is_gpu = accelerator_count() > 0
    n_warmup = _BENCH_WARMUP_GPU if is_gpu else _BENCH_WARMUP_CPU
    dev_label = f"GPU ({accelerator_architecture_name()})" if is_gpu else "CPU"

    print(
        f"\nDevice : {dev_label}  |  Warmup : {n_warmup} pass(es)", flush=True
    )
    print(f"{'─' * 52}", flush=True)

    # Mode 1: eager JIT compilation (default)
    os.environ.pop("MAX_USE_EAGER_INTERPRETER", None)
    _one_benchmark_pass(
        compiled_sampler,
        tokenizer,
        device,
        dtype,
        n_warmup=n_warmup,
        label="Eager JIT  (MAX_USE_EAGER_INTERPRETER unset)",
    )

    # Mode 2: eager interpreter
    os.environ["MAX_USE_EAGER_INTERPRETER"] = "1"
    _one_benchmark_pass(
        compiled_sampler,
        tokenizer,
        device,
        dtype,
        n_warmup=n_warmup,
        label="Interpreter (MAX_USE_EAGER_INTERPRETER=1)",
    )
    os.environ.pop("MAX_USE_EAGER_INTERPRETER", None)

    print(f"\n{'─' * 52}", flush=True)
    print(
        "Note: compiled transformer + logsoftmax are unaffected by the interpreter.",
        flush=True,
    )
    print(
        "Both modes use: compiled_sampler (GPU) + DLPack transfer + numpy Gumbel.",
        flush=True,
    )


# ANCHOR_END: benchmark


# ANCHOR: load_weights_and_run_model
def main() -> None:
    parser = argparse.ArgumentParser(description="MAX GPT-2 text generation")
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run timed benchmark instead of interactive generation",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Run single generation with this prompt and exit (non-interactive)",
    )
    args = parser.parse_args()

    dtype, device = defaults()
    print(f"Using device: {device}, dtype: {dtype}")

    # Load HuggingFace model
    torch_dtype = torch.bfloat16 if dtype == DType.bfloat16 else torch.float32
    hf_model = GPT2LMHeadModel.from_pretrained("gpt2", torch_dtype=torch_dtype)
    print(f"Loaded HuggingFace model:\n{hf_model}")

    config = GPT2Config()
    print(
        f"Model has {config.n_layer} layers, {config.n_head} heads,"
        f" {config.n_embd} embedding dim"
    )

    # 1. Build MAX model and load weights. `defaults()` resolves `device` to
    #    GPU when one is available; input tensors and compile types both use
    #    that device so everything stays on the same device without .to().
    #    HuggingFace GPT-2 Conv1D stores weights as [in, out]; MAX Linear
    #    expects [out, in], so pre-transpose before loading.
    print("Building model and loading weights...", flush=True)
    hf_state = hf_model.state_dict()
    transposed_state: dict[str, torch.Tensor] = {}
    for name, param in hf_state.items():
        if any(
            k in name for k in ["c_attn", "c_proj", "c_fc"]
        ) and name.endswith(".weight"):
            transposed_state[name] = param.T.contiguous()
        else:
            transposed_state[name] = param

    # F.lazy() defers all ops inside the block — random.normal in
    # Linear.__init__ / Embedding.__init__ is NEVER compiled or allocated.
    # load_state_dict replaces the lazy random tensors with the real HF
    # weights before they are ever realized.
    t0 = time.perf_counter()
    with F.lazy():
        max_model = MaxGPT2LMHeadModel(config)
        max_model.load_state_dict(transposed_state)
    print(
        f"  model init   : {(time.perf_counter() - t0) * 1e3:.0f} ms (lazy)",
        flush=True,
    )

    t0 = time.perf_counter()
    max_model.to(device)
    print(
        f"  to({device})  : {(time.perf_counter() - t0) * 1e3:.0f} ms",
        flush=True,
    )

    # 2. Wrap in compiled heads.
    sampling_head = GPT2SamplingHead(max_model)
    greedy_head = GPT2GreedyHead(max_model)

    token_type = TensorType(DType.int64, ("batch", "seqlen"), device=device)
    temp_type = TensorType(dtype, [], device=device)

    print("\nCompiling sampling model...", flush=True)
    t_compile_start = time.perf_counter()
    compiled_sampler = sampling_head.compile(token_type, temp_type)

    print("Compiling greedy model...", flush=True)
    compiled_greedy = greedy_head.compile(token_type)
    t_compile_end = time.perf_counter()
    print(f"Compile time: {t_compile_end - t_compile_start:.2f}s", flush=True)

    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    if args.benchmark:
        run_benchmark(compiled_sampler, tokenizer, device, dtype)
        return

    if args.prompt:
        generate_text(
            compiled_sampler,
            compiled_greedy,
            tokenizer,
            device,
            dtype,
            args.prompt,
            max_new_tokens=20,
            temperature=0.8,
            do_sample=True,
        )
        return

    # Interactive prompt loop
    print("\n" + "=" * 50)
    print("Model ready! Enter prompts to generate text.")
    print("Press Ctrl+C or type 'quit' to exit.")
    print("=" * 50 + "\n")

    try:
        while True:
            user_input = input("Enter your prompt: ").strip()

            if user_input.lower() in ["quit", "exit", "q"]:
                print("Exiting...")
                break

            if not user_input:
                print("Please enter a non-empty prompt.\n")
                continue

            print()
            generated_text = generate_text(
                compiled_sampler,
                compiled_greedy,
                tokenizer,
                device,
                dtype,
                user_input,
                max_new_tokens=50,
                temperature=0.8,
                do_sample=True,
            )
            print(f"\nGenerated text:\n{generated_text}\n")
            print("-" * 50 + "\n")

    except KeyboardInterrupt:
        print("\n\nExiting...")


# ANCHOR_END: load_weights_and_run_model

if __name__ == "__main__":
    main()

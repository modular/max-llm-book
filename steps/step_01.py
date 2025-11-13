"""
Step 01: Model Configuration

Implement the GPT-2 configuration dataclass that stores model hyperparameters.

Tasks:
1. Import dataclass from the dataclasses module
2. Add the @dataclass decorator to the GPT2Config class
3. Fill in the configuration values from HuggingFace GPT-2 model

Run: pixi run s01
"""

# 1. Import dataclass from the dataclasses module

# 2. Add the Python @dataclass decorator to the GPT2Config class


class GPT2Config:
    """GPT-2 configuration matching HuggingFace.

    Attributes:
        vocab_size: Size of the vocabulary.
        n_positions: Maximum sequence length.
        n_embd: Embedding dimension.
        n_layer: Number of transformer layers.
        n_head: Number of attention heads.
        n_inner: Inner dimension of feed-forward network (defaults to 4 * n_embd if None).
        layer_norm_epsilon: Epsilon for layer normalization.
    """

    # 3a. Run `pixi run huggingface` to access the model parameters from the Hugging Face `transformers` library
    # 3b. Alternately, read the values from GPT-2 model card: https://huggingface.co/openai-community/gpt2/blob/main/config.json
    # 4. Replace the None of the GPT2Config properties with the correct values
    vocab_size: int = None
    n_positions: int = None
    n_embd: int = None
    n_layer: int = None
    n_head: int = None
    n_inner: int = None  # Equal to 4 * n_embd
    layer_norm_epsilon: float = None

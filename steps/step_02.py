"""
Step 02: Feed-forward Network (MLP)

Implement the MLP used in each transformer block with GELU activation.

Tasks:
1. Import functional (as F), Tensor, Linear, and Module from MAX
2. Create c_fc linear layer (embedding to intermediate dimension)
3. Create c_proj linear layer (intermediate back to embedding dimension)
4. Apply c_fc transformation in forward pass
5. Apply GELU activation function
6. Apply c_proj transformation and return result

Run: pixi run s02
"""

# 1: Import the required modules from MAX
# TODO: Import functional module from max.experimental with the alias F
# https://docs.modular.com/max/api/python/experimental/functional

# TODO: Import Tensor from max.experimental.tensor
# https://docs.modular.com/max/api/python/experimental/tensor.Tensor

# TODO: Import Linear and Module from max.nn.module_v3
# https://docs.modular.com/max/api/python/nn/module_v3

from step_01 import GPT2Config


class GPT2MLP(Module):
    """Feed-forward network matching HuggingFace GPT-2 structure.

    Args:
        intermediate_size: Size of the intermediate layer.
        config: GPT-2 configuration.
    """

    def __init__(self, intermediate_size: int, config: GPT2Config):
        super().__init__()
        embed_dim = config.n_embd

        # 2: Create the first linear layer (embedding to intermediate)
        # TODO: Create self.c_fc as a Linear layer from embed_dim to intermediate_size with bias=True
        # https://docs.modular.com/max/api/python/nn/module_v3#max.nn.module_v3.Linear
        # Hint: This is the expansion layer in the MLP
        self.c_fc = None

        # 3: Create the second linear layer (intermediate back to embedding)
        # TODO: Create self.c_proj as a Linear layer from intermediate_size to embed_dim with bias=True
        # https://docs.modular.com/max/api/python/nn/module_v3#max.nn.module_v3.Linear
        # Hint: This is the projection layer that brings us back to the embedding dimension
        self.c_proj = None

    def __call__(self, hidden_states: Tensor) -> Tensor:
        """Apply feed-forward network.

        Args:
            hidden_states: Input hidden states.

        Returns:
            MLP output.
        """
        # 4: Apply the first linear transformation
        # TODO: Apply self.c_fc to hidden_states
        # Hint: This expands the hidden dimension to the intermediate size
        hidden_states = None

        # 5: Apply GELU activation function
        # TODO: Use F.gelu() with hidden_states and approximate="tanh"
        # https://docs.modular.com/max/api/python/experimental/functional#max.experimental.functional.gelu
        # Hint: GELU is the non-linear activation used in GPT-2's MLP
        hidden_states = None

        # 6: Apply the second linear transformation and return
        # TODO: Apply self.c_proj to hidden_states and return the result
        # Hint: This projects back to the embedding dimension
        return None

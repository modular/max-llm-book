# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""
Solution for Step 08: Language Model Head

This module adds the final projection layer that converts hidden states
to vocabulary logits for predicting the next token.
"""

from max.experimental.nn import Linear, Module
from max.experimental.tensor import Tensor
from step_01 import GPT2Config
from step_07 import MaxGPT2Model


class MaxGPT2LMHeadModel(Module):  # type: ignore[type-arg]
    """Complete GPT-2 model with language modeling head.

    This is the full model that can be used for text generation.
    """

    def __init__(self, config: GPT2Config) -> None:
        """Initialize GPT-2 with LM head.

        Args:
            config: GPT2Config containing model hyperparameters
        """
        super().__init__()

        self.config = config
        # The transformer (embeddings + blocks + final norm)
        self.transformer = MaxGPT2Model(config)
        # Language modeling head (hidden states -> vocabulary logits)
        self.lm_head = Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, input_ids: Tensor) -> Tensor:
        """Forward pass through transformer and LM head.

        Args:
            input_ids: Token IDs, shape [batch, seq_length]

        Returns:
            Logits over vocabulary, shape [batch, seq_length, vocab_size]
        """
        # Get hidden states from transformer
        hidden_states = self.transformer(input_ids)

        # Project to vocabulary logits
        logits = self.lm_head(hidden_states)

        return logits

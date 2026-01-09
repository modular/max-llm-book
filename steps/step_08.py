"""
Step 08: Language Model Head

Add the final projection layer that converts hidden states to vocabulary logits.

Tasks:
1. Import Linear, Module, and previous components
2. Create transformer and lm_head layers
3. Implement forward pass: transformer -> lm_head

Run: pixi run s08
"""

# TODO: Import required modules
# Hint: You'll need Linear and Module from max.nn.module_v3

from step_01 import GPT2Config
from step_07 import MaxGPT2Model

class MaxGPT2LMHeadModel(Module):
    """Complete GPT-2 model with language modeling head."""

    def __init__(self, config: GPT2Config):
        """Initialize GPT-2 with LM head.

        Args:
            config: GPT2Config containing model hyperparameters
        """
        super().__init__()

        self.config = config

        # TODO: Create the transformer
        # Hint: Use MaxGPT2Model(config)
        self.transformer = None

        # TODO: Create language modeling head
        # Hint: Use Linear(config.n_embd, config.vocab_size, bias=False)
        # Projects from hidden dimension to vocabulary size
        self.lm_head = None

    def __call__(self, input_ids):
        """Forward pass through transformer and LM head.

        Args:
            input_ids: Token IDs, shape [batch, seq_length]

        Returns:
            Logits over vocabulary, shape [batch, seq_length, vocab_size]
        """
        # TODO: Get hidden states from transformer
        # Hint: hidden_states = self.transformer(input_ids)
        pass

        # TODO: Project to vocabulary logits
        # Hint: logits = self.lm_head(hidden_states)
        pass

        # TODO: Return logits
        return None

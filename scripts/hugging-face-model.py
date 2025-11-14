from transformers import GPT2LMHeadModel


def get_gpt2_config():
    # Load HuggingFace model for weight transfer
    print("Loading HuggingFace GPT-2 model...")
    torch_model = GPT2LMHeadModel.from_pretrained("gpt2")
    print(
        # Print model configuration to match GPT2Config
        "HuggingFace model configuration:\n"
        # vocab_size - size of the vocabulary
        f"vocab_size: {torch_model.transformer.wte.num_embeddings}\n"
        # n_positions - maximum sequence length
        f"n_positions: {torch_model.transformer.wpe.num_embeddings}\n"
        # n_embd - embedding dimension
        f"n_embd: {torch_model.transformer.wte.embedding_dim}\n"
        # n_layer - number of transformer blocks
        f"n_layer: {len(torch_model.transformer.h)}\n"
        # n_head - number of attention heads
        f"n_head: {torch_model.transformer.h[0].attn.num_heads}\n"
        # n_inner - inner dimension of MLP (feed-forward network)
        f"n_inner: {torch_model.transformer.h[0].mlp.c_fc.nf}\n"
        # layer_norm_epsilon - epsilon for layer normalization
        f"layer_norm_epsilon: {torch_model.transformer.ln_f.eps}\n"
    )
    # print(torch_model)


if __name__ == "__main__":
    get_gpt2_config()

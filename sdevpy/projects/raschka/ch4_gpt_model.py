# from importlib.metadata import version
# import torch
# import numpy as np
# from sdevpy.llms.attention import SelfAttentionV1, SelfAttentionV2, CausalAttention
# from sdevpy.llms.attention import MultiHeadAttentionWrapper, MultiHeadAttention

# print("pytorch version: ", torch.__version__)


GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers (number of transformer blocks)
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # Query-Key-Value bias
}

# print("<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>")
# print("<><><><><><><><> Non-trainable self-attention <><><><><><><><><><><><><><><><>")
# print("<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>\n")

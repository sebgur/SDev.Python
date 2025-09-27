import re
from importlib.metadata import version
import torch
import numpy as np
# from torch.utils.data import Dataset, DataLoader
# from sdevpy.llms.tokenizers import SimpleTokenizerV1, SimpleTokenizerV2
# from sdevpy.projects.raschka import torch_datasetloader as tdsl

print("pytorch version: ", torch.__version__)

# Originally, translators were based on RNN networks using the encoder-decoder architecture.
# This had issues with long-term memory.
# Modern models are based on the Attention mechanism, with the Transformer architecture.

# A context vector is an enriched embedding vector. It is enriched here through the Attention
# mechanism, by including information from all the other token embeddings in the sequence.

print("<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>")
print("<><><><><><><><> Non-trainable self-attention <><><><><><><><><><><><><><><><>")
print("<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>\n")
# Define input sequence, output_dim = 3
inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

sequence_size = inputs.shape[0]
print(f"Sequence size: {sequence_size}", "\n")
embed_dim = inputs[0].shape
print(f"Embedding dimension: {embed_dim}", "\n")

# Calculate raw attention scores
# The original token x of which we're calculating the context vector is called the "query".
# The attention scores are the dot product between the query element and the other elements
# in the sequence.
query = inputs[1]
attn_scores_2 = torch.empty(sequence_size)
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)

print("Attention scores for query x2:\n", attn_scores_2, "\n")

# Normalized attention scores (attention weights)
attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
print("Attention weights (simple normalization):\n", attn_weights_2_tmp, "\n")
print("Attention weight sum:\n", attn_weights_2_tmp.sum(), "\n")

# In practice the softmax function is more used for normalization as it has better properties
# for extreme values and gradient. That is:
# w_i = exp(x_i) / \Sum_j exp(x_j)
def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)

attn_weights_2_naive = softmax_naive(attn_scores_2)
print("Attention weights (naive softmax):\n", attn_weights_2_naive, "\n")
print("Attention weight sum:\n", attn_weights_2_naive.sum(), "\n")

# Using more optimized softmax version in PyTorch
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print("Attention weights (PyTorch softmax):\n", attn_weights_2, "\n")
print("Attention weight sum:\n", attn_weights_2.sum(), "\n")

# Calculate the context vector for query 2
context_vec_2 = torch.zeros(embed_dim)
for i in range(sequence_size):
    context_vec_2 += inputs[i] * attn_weights_2[i]

print("Convext vector query 2:\n", context_vec_2, "\n")

# Compute attention weights for all input tokens
attn_scores = torch.empty(sequence_size, sequence_size)
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)

print("All attention scores\n", attn_scores, "\n")

# For better performance, we should use matrix multiplication
attn_scores = inputs @ inputs.T
print("All attention scores (matrix mult)\n", attn_scores, "\n")

attn_weights = torch.softmax(attn_scores, dim=-1) # Same as dim = 1, so inside each vector i.e. per row
print("All attention weights\n", attn_weights, "\n")

# Verify normalization sums on rows
print("All row sums: ", attn_weights.sum(dim=-1), "\n")

print(inputs)

# Compute all context vectors
print(attn_weights.shape)
print(inputs.shape)
all_context_vecs = attn_weights @ inputs
print("All context vectors\n", all_context_vecs, "\n")


print("<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>")
print("<><><><><><><><> Scaled Dot-Product Attention <><><><><><><><><><><><><><><><>")
print("<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>\n")
x_2 = inputs[1]
d_in = inputs.shape[1] # Embedding dimension
d_out = 2 # Output embedding size (in practice, taken the same as d_in)

torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

query_2 = x_2 @ W_query
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value
print("Query 2\n", query_2, "\n")

keys = inputs @ W_key
values = inputs @ W_value
print("keys.shape: ", keys.shape)
print("values.shape: ", values.shape)

# Calculate one attention score (2-2)
keys_2 = keys[1]
attn_score_22 = query_2.dot(keys_2) # Regular dot product
print("Attention score 22: ", attn_score_22, "\n")

# Calculate all attention scores for given query (2)
attn_scores_2 = query_2 @ keys.T
print("Attention scores for 2\n", attn_scores_2, "\n")

# Calculate all attention weights (by scaling) for given query (2)
# The scaling is done by the square root of the embedding dimension, and that is
# the reason for naming this mechanism "Scaled Dot-Product Attention". This scaling
# is done to avoid small gradients.
d_k = keys.shape[-1]
print(d_k)
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
print("Attention weights for 2\n", attn_weights_2, "\n")

# Calculate the context vector by multiplying the weights with the values
context_vec_2 = attn_weights_2 @ values
print("Context vector for 2\n", context_vec_2, "\n")


from importlib.metadata import version
import torch
import torch.nn as nn
import tiktoken
import matplotlib.pyplot as plt
from sdevpy.llms.gpt import DummyGPTModel, LayerNorm, GELU, FeedForward

print("tiktoken version:", version("tiktoken"))
print("pytorch version: ", torch.__version__)


GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers (number of transformer blocks)
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # Query-Key-Value bias
}

print("<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>")
print("<><><><><><><><> Structure Overview <><><><><><><><><><><><><><><><><><><><><>")
print("<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>\n")
print("<><><><> Tokenize sample text <><><><>")
tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"
batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)
print("Batch\n", batch, "\n")

print("Initialize GPT model")
torch.manual_seed(123)
model = DummyGPTModel(GPT_CONFIG_124M)
logits = model(batch)
print("Output shape: ", logits.shape)
print("Logits\n", logits, "\n")

print("<><><><> Layer Normalization <><><><>")
torch.manual_seed(123)
batch_example = torch.randn(2, 5)
layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
out = layer(batch_example)
print("Example batch\n", batch_example, "\n")
print("NN output before normalization\n", out, "\n")
mean = out.mean(dim=-1, keepdim=True)
var = out.var(dim=-1, keepdim=True)
print("Mean after normalization\n", mean, "\n")
print("Variance after normalization\n", var, "\n")

# Normalize
print("<><><><> Normalize by simple formulas")
torch.set_printoptions(sci_mode=False) # Turn off scientific notation
out_norm = (out - mean) / torch.sqrt(var)
mean = out_norm.mean(dim=-1, keepdim=True)
var = out_norm.var(dim=-1, keepdim=True)
print("Normalized\n", out_norm, "\n")
print("Mean after normalization\n", mean, "\n")
print("Variance after normalization\n", var, "\n")

print("<><><><> Normalize using LayerNorm")
ln = LayerNorm(emb_dim=5)
out_ln = ln(batch_example)
mean = out_ln.mean(dim=-1, keepdim=True)
var = out_ln.var(dim=-1, keepdim=True, unbiased=False)
print("Normalized\n", out_ln, "\n")
print("Mean after LayerNorm\n", mean, "\n")
print("Variance after LayerNorm\n", var, "\n")

# print("<><><><> GELU")
# gelu, relu = GELU(), nn.ReLU()
# x = torch.linspace(-3, 3, 100)
# y_gelu, y_relu = gelu(x), relu(x)
# plt.figure(figsize=(8, 3))
# for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "ReLU"]), 1):
#     plt.subplot(1, 2, i)
#     plt.plot(x, y)
#     plt.title(f"{label} activation function")
#     plt.xlabel("x")
#     plt.ylabel(f"{label}(x)")
#     plt.grid(True)
# plt.tight_layout()
# plt.show()

print("<><><><> Feed-Forward")
ffn = FeedForward(GPT_CONFIG_124M)
x = torch.rand(2, 3, 768)
out = ffn(x)
print(f"Output shape: {out.shape}")




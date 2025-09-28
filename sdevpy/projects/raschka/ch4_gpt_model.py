from importlib.metadata import version
import torch
import torch.nn as nn
import tiktoken
import matplotlib.pyplot as plt
from sdevpy.llms.gpt import DummyGPTModel, LayerNorm, FeedForward, TransformerBlock, GPTModel
from sdevpy.projects.raschka import raschka_dnn

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


print("<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>")
print("<><><><><><><><> Feed-Forward and Shorctut Layers <><><><><><><><><><><><><><>")
print("<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>\n")
ffn = FeedForward(GPT_CONFIG_124M)
x = torch.rand(2, 3, 768)
out = ffn(x)
print(f"Output shape: {out.shape}")

print("<><><><> Shortcut Connections")
# Also known as skip or residual connections
layer_sizes = [3, 3, 3, 3, 3, 1]
sample_input = torch.tensor([[1., 0., -1.]])
torch.manual_seed(123)
model_without_shortcut = raschka_dnn.ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=False)
print("Model without shortcuts\n", model_without_shortcut, "\n")
raschka_dnn.print_gradients(model_without_shortcut, sample_input)

model_with_shortcut = raschka_dnn.ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=True)
print("Model with shortcuts\n", model_with_shortcut, "\n")
raschka_dnn.print_gradients(model_with_shortcut, sample_input)


print("<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>")
print("<><><><><><><><> Transformer Block and GPT  <><><><><><><><><><><><><><><><><>")
print("<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>\n")
torch.manual_seed(123)
x = torch.rand(2, 4, 768)
print("<><><><> Transformer Block")
block = TransformerBlock(GPT_CONFIG_124M)
output = block(x)

print("Input shape: ", x.shape)
print("Output shape: ", output.shape)

print("<><><><> GPT Model")
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
out = model(batch)
print("Input shape: ", batch.shape)
print("Input batch\n", batch, "\n")
print("Output shape: ", out.shape)
print("Output batch\n", out, "\n")

# Number of parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters {total_params:,}")

print("Token embedding layer shape: ", model.tok_emb.weight.shape)
print("Output layer shape: ", model.out_head.weight.shape)

# Number parameters when using weight tying
total_params_gpt2 = (total_params- sum(p.numel() for p in model.out_head.parameters()))
print(f"Number of trainable parameters with weight tying: {total_params_gpt2:,}")

# Memory requirement
total_size_bytes = total_params * 4
total_size_mb = total_size_bytes / (1024 * 1024)
print(f"Total size of the model: {total_size_mb:.2f} MB")


print("<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>")
print("<><><><><><><><> Generating Text  <><><><><><><><><><><><><><><><><>")
print("<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>\n")

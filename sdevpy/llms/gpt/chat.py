import torch
import tiktoken
from sdevpy.machinelearning.gpt import GPTModel
import sdevpy.machinelearning.llms.textgen as tg


GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 256, # Context length
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers (number of transformer blocks)
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # Query-Key-Value bias
}

file = r"C:\\temp\\llms\\model-save.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(file, map_location=device)
model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(checkpoint["model_state_dict"])
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
# model.train();

tokenizer = tiktoken.get_encoding("gpt2")

start_text = "Every effort moves you"

start_tensor = tg.text_to_token_ids(start_text, tokenizer)
token_ids = tg.generate_text_simple(model=model, idx=start_tensor, max_new_tokens=50,
                                    context_size=GPT_CONFIG_124M["context_length"])

print("Input: " + start_text)
print("Output: " + tg.token_ids_to_text(token_ids, tokenizer).replace(start_text, ''))

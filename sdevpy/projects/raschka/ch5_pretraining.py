from importlib.metadata import version
import torch
import tiktoken
from sdevpy.llms.gpt import GPTModel
import sdevpy.llms.textgen as tg
from sdevpy.projects.raschka import raschka_datasetloader as ds
from sdevpy.llms.training import calc_loss_loader, train_model_simple, plot_losses

print("tiktoken version:", version("tiktoken"))
print("pytorch version: ", torch.__version__)

# Chp 5.1, page 141, reference to large scale dataset of public domain books.
# Chp 5.2, page 146, learn about learning rate warmup, cosine annealing and gradient clipping

GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 256, # Context length
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers (number of transformer blocks)
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # Query-Key-Value bias
}

print("<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>")
print("<><><><><><><><> Using GPT  <><><><><><><><><><><><><><><><><><><><><><><><><>")
print("<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>\n")
print("<><><><> Generate text (untrained) <><><><>")
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.eval()

start_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")
start_tensor = tg.text_to_token_ids(start_context, tokenizer)

token_ids = tg.generate_text_simple(model=model, idx=start_tensor, max_new_tokens=10,
                                    context_size=GPT_CONFIG_124M["context_length"])

print("Output text:\n", tg.token_ids_to_text(token_ids, tokenizer))

print("<><><><> Text generation loss <><><><>")
start_context = ["every effort moves", "I really like"]
inputs = torch.tensor([tokenizer.encode(x) for x in start_context])
print("Inputs\n", inputs, "\n")

end_context = [" effort moves you", " really like chocolate"]
targets = torch.tensor([tokenizer.encode(x) for x in end_context])
print("Targets\n", targets, "\n")

# Use the model to predict
with torch.no_grad():
    logits = model(inputs)

print(f"Logit shape: {logits.shape}")
probas = torch.softmax(logits, dim=-1)
print(f"Proba shape: {probas.shape}")

token_ids = torch.argmax(probas, dim=-1, keepdim=True)
print("Most prob token IDs\n", token_ids, "\n")

print(f"Targets batch 1: {tg.token_ids_to_text(targets[0], tokenizer)}")
print(f"Output batch 1: {tg.token_ids_to_text(token_ids[0].flatten(), tokenizer)}")

text_idx = 0 # Index in batch
target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]]
print("Probas text 1\n", target_probas_1, "\n")

text_idx = 1 # Index in batch
target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]]
print("Probas text 2\n", target_probas_2, "\n")

# Calculate log of target probas
log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
print("Log target probas\n", log_probas, "\n")
avg_log_probas = torch.mean(log_probas)
print(f"Average log proba\n", avg_log_probas, "\n")
neg_avg_log_probas = avg_log_probas * -1
print(f"Negative average log proba (Cross-Entropy)\n", neg_avg_log_probas, "\n")

# Using PyTorch's cross-entropy function
print("Logits shape: ", logits.shape)
print("Targets shape: ", targets.shape)
logits_flat = logits.flatten(0, 1)
targets_flat = targets.flatten()
print("Flatten logit shape: ", logits_flat.shape)
print("Flatten target shape: ", targets_flat.shape)

loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
print("Loss using Cross-Entropy: ", loss, "\n")
print("Perplexity: ", torch.exp(loss), "\n")


print("<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>")
print("<><><><><><><><> Training <><><><><><><><><><><><><><><><><><><><><><><><><><>")
print("<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>\n")
print("<><><><> Loss on the entire sets <><><><>")
file = "datasets/llms/the-verdict.txt"
with open(file, "r", encoding="utf-8") as f:
    text_data = f.read()

total_characters = len(text_data)
total_tokens = len(tokenizer.encode(text_data))
print(f"Characters: {total_characters}")
print(f"Tokens: {total_tokens}")

print("<><> Create dataset loaders")
train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

torch.manual_seed(123)
train_loader = ds.create_dataloader_v1(train_data, batch_size=2,
                                       max_length=GPT_CONFIG_124M["context_length"],
                                       stride=GPT_CONFIG_124M["context_length"],
                                       drop_last=True, shuffle=True, num_workers=0)

val_loader = ds.create_dataloader_v1(val_data, batch_size=2,
                                     max_length=GPT_CONFIG_124M["context_length"],
                                     stride=GPT_CONFIG_124M["context_length"],
                                     drop_last=False, shuffle=False, num_workers=0)

print("Train loader:")
for x, y in train_loader:
    print(x.shape, y.shape)

print("\nValidation loader:")
for x, y in val_loader:
    print(x.shape, y.shape)

print("<><> Calculate losses")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
with torch.no_grad():
    train_loss = calc_loss_loader(train_loader, model, device)
    val_loss = calc_loss_loader(val_loader, model, device)

print("Training loss: ", train_loss)
print("Validation loss: ", val_loss)

print("<><> Simple training loop")
torch.manual_seed(123)
start_text = "Every effort moves you"
print("Starting text: " + start_text)
model = GPTModel(GPT_CONFIG_124M)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
num_epochs = 10
train_losses, val_losses, tokens_seen = train_model_simple(model, train_loader, val_loader, optimizer,
                                                           device, num_epochs=num_epochs, eval_freq=5,
                                                           eval_iter=5, start_context=start_text,
                                                           tokenizer=tokenizer)

# file_save = "model-save.pth"
# torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),}, file_save)

epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)





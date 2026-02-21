from importlib.metadata import version
import torch
import tiktoken
import matplotlib.pyplot as plt
from sdevpy.machinelearning.llms.gpt import GPTModel
import sdevpy.machinelearning.llms.textgen as tg
from sdevpy.projects.raschka import raschka_datasetloader as ds
from sdevpy.machinelearning.llms.training import calc_loss_loader, train_model_simple, plot_losses

print("tiktoken version:", version("tiktoken"))
print("pytorch version: ", torch.__version__)

# Chp 5.1, page 141, reference to large scale dataset of public domain books.
# Chp 5.2, page 146, learn about learning rate warmup, cosine annealing and gradient clipping

# ToDo: would be interesting to understand, after training, what are the actual words
# in the rest of the context output by the model. That is, we only use the last element
# in the sequence and interpret it as the predicted word, but in reality the model
# outputs a whole sequence, out of which we only use the last token. But what do the
# other ones represent?

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

# ########################################## TRAIN ##############################################
# print("<><> Simple training loop")
# torch.manual_seed(123)
# start_text = "Every effort moves you"
# print("Starting text: " + start_text)
# model = GPTModel(GPT_CONFIG_124M)
# model.to(device)
# optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
# num_epochs = 10
# train_losses, val_losses, tokens_seen = train_model_simple(model, train_loader, val_loader, optimizer,
#                                                            device, num_epochs=num_epochs, eval_freq=5,
#                                                            eval_iter=5, start_context=start_text,
#                                                            tokenizer=tokenizer)

# # file_save = "model-save.pth"
# # torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),}, file_save)

# # epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
# # plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
# ##############################################################################################

# ############### LOAD SAVED MODEL #############################################################
print("<><> Load saved model")
file = r"C:\\temp\\llms\\model-save.pth"
# torch.manual_seed(123)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(file, map_location=device)
model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(checkpoint["model_state_dict"])
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
# # model.train();

# tokenizer = tiktoken.get_encoding("gpt2")

# start_text = "Every effort moves you"

# start_tensor = tg.text_to_token_ids(start_text, tokenizer)
model.eval()
# token_ids = tg.generate_text_simple(model=model, idx=start_tensor, max_new_tokens=25,
#                                     context_size=GPT_CONFIG_124M["context_length"])

# print("Input: " + start_text)
# print("Output: " + tg.token_ids_to_text(token_ids, tokenizer).replace(start_text, ''))
################################################################################################

print("<><> Temperature Scaling")
vocab = {"closer": 0, "every": 1, "effort": 2, "forward": 3, "inches": 4, "moves": 5,
         "pizza": 6, "toward": 7, "you": 8, }
inverse_vocab = {v: k for k, v in vocab.items()}

next_token_logits = torch.tensor([4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79])
print("Logits: ", next_token_logits)
probas = torch.softmax(next_token_logits, dim=0)
print("Probas: ", probas)
print("ArgMax: ", torch.argmax(probas))
next_token_id = torch.argmax(probas).item()
print("Next token ID: ", next_token_id)
print(inverse_vocab[next_token_id])

# Probabilitic method using multinomial
torch.manual_seed(123)
next_token_id = torch.multinomial(probas, num_samples=1).item()
print(inverse_vocab[next_token_id])

def print_sampled_tokens(probas):
    torch.manual_seed(123)
    sample = [torch.multinomial(probas, num_samples=1).item() for i in range(1000)]
    sampled_ids = torch.bincount(torch.tensor(sample))
    for i, freq in enumerate(sampled_ids):
        print(f"{freq} x {inverse_vocab[i]}")

print_sampled_tokens(probas)

# Temperature scaling
def softmax_with_temperature(logits, temperature):
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=0)

temperatures = [1, 0.1, 5]
scaled_probas = [softmax_with_temperature(next_token_logits, T) for T in temperatures]
x = torch.arange(len(vocab))
bar_width = 0.15
fig, ax = plt.subplots(figsize=(5, 3))
for i, T in enumerate(temperatures):
    rects = ax.bar(x + i * bar_width, scaled_probas[i], bar_width, label=f'Temperature={T}')
ax.set_ylabel('Probability')
ax.set_xticks(x)
ax.set_xticklabels(vocab.keys(), rotation=90)
ax.legend()
plt.tight_layout()
plt.show()

# Top-k sampling
top_k = 3
top_logits, top_pos = torch.topk(next_token_logits, top_k)
print("Top logits: ", top_logits)
print("Top positions: ", top_pos)

new_logits = torch.where(condition=next_token_logits < top_logits[-1],
                         input=torch.tensor(float('-inf')),other=next_token_logits)
print(new_logits)

topk_probas = torch.softmax(new_logits, dim=0)
print(topk_probas)


# Better text generation
torch.manual_seed(123)
start_text = "Every effort moves you"
token_ids = tg.generate(model=model, idx=tg.text_to_token_ids(start_text, tokenizer),
                        max_new_tokens=15, context_size=GPT_CONFIG_124M["context_length"],
                        top_k=25, temperature=1.4)

print("Output text:\n", tg.token_ids_to_text(token_ids, tokenizer))

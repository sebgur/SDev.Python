import gc
import torch, tiktoken
from sdevpy.llms.gpt import gpt
from sdevpy.llms.gpt import datasetloader as dsl
from sdevpy.llms.gpt.training import train_gpt_model, plot_losses
from sdevpy.utilities.tools import workpath


####################### Runtime configuration #########################################################################
model_name = "testgpt"
start_version = 2 # When start_version = 0, the model is started from scratch
start_model_name = f"{model_name}-{start_version:03d}"
end_model_name = f"{model_name}-{start_version + 1:03d}"
print(f"Start model: {start_model_name}")
print(f"End model: {end_model_name}")

# Training dataset
dataset_name = "the-verdict"

# Model config
gpt_config = {"vocab_size": 50257, "context_length": 256, "emb_dim": 768, "n_heads": 4, "n_layers": 6,
              "drop_rate": 0.1, "qkv_bias": False}
# # 124M
# gpt_config = {"vocab_size": 50257, "context_length": 256, "emb_dim": 768, "n_heads": 12, "n_layers": 12,
#               "drop_rate": 0.1, "qkv_bias": False}

epochs = 10
batch_size = 2
train_ratio = 0.90
init_lr = 0.0004

####################### Training ######################################################################################
# Initial setup
project_path = workpath() / "llms"
model_path = project_path / "models"
data_path = project_path / "datasets"
data_file = data_path / (dataset_name + ".txt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(123)
tokenizer = tiktoken.get_encoding("gpt2")
ctx_length = gpt_config["context_length"]

# Read the dataset
print(f"Read training text data from file: {data_file}")
with open(data_file, encoding="utf-8") as f:
    text_data = f.read()

print(f"Characters: {len(text_data):,}")
print(f"Tokens: {len(tokenizer.encode(text_data)):,}")

print("Create dataset loaders")
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

# torch.manual_seed(123)
train_loader = dsl.create_dataloader(train_data, batch_size=batch_size, max_length=ctx_length, stride=ctx_length,
                                     drop_last=True, shuffle=True, num_workers=0)

val_loader = dsl.create_dataloader(val_data, batch_size=batch_size, max_length=ctx_length, stride=ctx_length,
                                   drop_last=False, shuffle=False, num_workers=0)

print("<><><><><><><><> Start training <><><><><><><><>")
start_text = "Every effort moves you"
print("Test start: " + start_text)

# Initialize model
if start_version == 0:
    model = gpt.GptModule(gpt_config)
else:
    # Read from previous
    start_model_path = model_path / start_model_name
    model = gpt.load_model_from_path(start_model_path, device=device)

model.to(device)

# Training parameters
optimizer = torch.optim.AdamW(model.parameters(), lr=init_lr, weight_decay=0.1)

train_losses, val_losses, tokens_seen = train_gpt_model(model, train_loader, val_loader, optimizer,
                                                        device, num_epochs=epochs, eval_freq=5,
                                                        eval_iter=5, start_context=start_text,
                                                        tokenizer=tokenizer)

# Save pretrained model to new version
end_model_path = model_path / end_model_name
gpt.save_model_to_path(model, gpt_config, end_model_path)

# Plot diagnostics
epochs_tensor = torch.linspace(0, epochs, len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

# Unload
del model
gc.collect()

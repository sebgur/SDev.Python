import os, gc
from pathlib import Path
import torch
import tiktoken
from sdevpy.llms.gpt import gpt
from sdevpy.llms.gpt import textgen as tg
from sdevpy.utilities import jsonmanager as jsm


model_size = "1558M" # 124M, 355M, 774M, 1558M
project_path = Path(os.environ.get('SDEVPY_DATA', Path.home() / 'sdevpy'))
max_sentences = 2

project_path = project_path / "llms" / "gpt"

torch.manual_seed(123)
tokenizer = tiktoken.get_encoding("gpt2")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Stopping characters
period = tokenizer.encode(".")[0]
exclamation = tokenizer.encode("!")[0]
question = tokenizer.encode("?")[0]
eos_tokens = [period, exclamation, question]
eot_tokens = [tokenizer.eot_token]

# Load model from pretrained
model_path = project_path / f"gpt2-{model_size}"
weight_file = model_path / "weights.pth"
config_file = model_path / "config.json"
model_config = jsm.deserialize(config_file)
model = gpt.GPTModel(model_config)
checkpoint = torch.load(weight_file, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
context_length = model_config["context_length"]

# Initialize model
model.to(device)

print("Generating text...")

# Choose token generator
token_gen = tg.NextTokenGenerator(top_k=15, temperature=1.5)

# # Initialize chat generator
chat_gen = tg.ChatGenerator(device, model, tokenizer, context_length, token_gen,
                            max_new_tokens=50, max_sentences=max_sentences)

# Iteration 1
start_text = "Why is the sky blue?"
print()
print("Input text:\n", start_text)
print()
end_text = chat_gen.end_text(start_text)
print("Output text:\n", tg.format_answer(start_text, end_text))

# Iteration 2
new_text = "Tell me a joke"
start_text = end_text + "\n" + new_text
print()
print("Input text:\n", new_text)
print()
end_text = chat_gen.end_text(start_text)
print("Output text:\n", tg.format_answer(start_text, end_text))

# Unload
del model
gc.collect()

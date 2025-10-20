import os
import torch
import re
import tiktoken
from sdevpy.llms import gpt
from sdevpy.llms import textgen as tg
# Develop script for json instruction on raw GPT2 model
# Develop script for pre-training on json instructions
# Develop script for LORA fine-tuning on json instructions
# Learn how to download/load Llama, Store Llama in OneDrive
# Test Llama on json instruction
# Test fine-tuning Llama on json instruction
# Test code on GPU-enabled platforms


if __name__ == "__main__":
    model_size = "355M" # 124M, 355M, 774M, 1558M
    max_sentences = 4

    torch.manual_seed(123)
    tokenizer = tiktoken.get_encoding("gpt2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Stopping characters
    period = tokenizer.encode(".")[0]
    exclamation = tokenizer.encode("!")[0]
    question = tokenizer.encode("?")[0]
    eos_tokens = [period, exclamation, question]
    eot_tokens = [tokenizer.eot_token]

    # Load parameters
    model_folder = r"C:\\SDev.Finance\\OneDrive\\LLM\\models\\gpt2"
    model_folder = os.path.join(model_folder, model_size)

    settings, params = gpt.load_gpt2(model_folder)
    # print("Settings: ", settings)
    # print("Param dict keys: ", params.keys())

    # Create model
    GPT_CONFIG = {"vocab_size": settings['n_vocab'], "context_length": settings['n_ctx'],
                  "emb_dim": settings['n_embd'], "n_heads": settings['n_head'],
                  "n_layers": settings['n_layer'], "drop_rate": 0.1, "qkv_bias": True}

    model = gpt.GPTModel(GPT_CONFIG)
    model.eval()

    # Load parameters into model
    print("Loading weights for GPT-2 model size: " + model_size)
    gpt.load_weights(model, params)
    print("Done loading weights!")
    print()

    # Initialize model
    model.to(device)

    print("Generating text...")

    # Feed input
    start_text = "The red house was too far away."
    print("Input text:\n", start_text)
    print()

    token_gen = tg.NextTokenGenerator(top_k=15, temperature=1.5)
    # token_gen = tg.NextTokenGenerator()
    # token_gen = tg.SimpleTokenGenerator()

    tokens = tg.generate(model, tg.text_to_tokens(start_text, tokenizer).to(device),
                         max_new_tokens=250, context_size=GPT_CONFIG["context_length"],
                         token_generator=token_gen, eos_ids=eos_tokens, max_sentences=max_sentences,
                         eot_ids=eot_tokens)

    print("Output text:\n", tg.format_answer(start_text, tg.tokens_to_text(tokens, tokenizer)))

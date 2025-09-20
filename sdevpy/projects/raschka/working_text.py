import re
from importlib.metadata import version
from sdevpy.llms.tokenizers import SimpleTokenizerV1, SimpleTokenizerV2
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

print("tiktoken version:", version("tiktoken"))
print("pytorch version: ", torch.__version__)

############ Simple regex/parsing examples ################################################
text = "Hello, world. This, is a test."
# print("Original text\n", text)
result = re.split(r'(\s)', text)
# print("Split according to white spaces\n", result)

result = re.split(r'([,.]|\s)', text)
# print("Split punctation\n", result)

# Remove white spaces
result = [item.strip() for item in result if item.strip()] # Remove trailing whitespaces, tabs, newlines and returns
# print("Remove white spaces\n", result)

# Handle more characters
text = "Hello, world. Is this -- a test?"
# print("Original text\n", text)
result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
result = [item.strip() for item in result if item.strip()]
# print("Splitting more characters\n", result)

############ Build vocabulary ################################################
# Read text file
file = "datasets/llms/the-verdict.txt"
with open(file, "r", encoding="utf-8") as f:
    raw_text = f.read()

print(f"Number of characters in text: {len(raw_text)}")
#print("Head\n", file_data[:99])

# Split whole file into tokens
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(f"Number of tokens in text: {len(preprocessed)}")
# print(preprocessed[:12])

# Build unique set of tokens and sort
unique_tokens = set(preprocessed)
all_words = sorted(unique_tokens)
vocab_size = len(all_words)
print(f"Vocabulary size (number unique tokens): {vocab_size}")

# Extract vocabulary dictionary <token, tokenId>
vocab = {token:integer for integer, token in enumerate(all_words)} # Nice, didn't know we could do that
# for i, item in enumerate(vocab.items()):
#     print(item)
#     if i >= 12:
#         break

############ Use tokenizer classes ################################################
# Use tokenizer class
tokenizer = SimpleTokenizerV1(vocab)

known_text = """"It's the last he painted, you know,"
 Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(known_text)
dctxt = tokenizer.decode(ids)
# print("Use tokenizer class to encode\n", ids)
# print("Use tokenizer class to decode\n", dctxt) # Seb: small bugs with space after apostrophe

# Try to encode a text with a word that's not part of the vocabulary
# unk_text = "Hello, do you like tea?"
# print(tokenizer.encode(unk_text))

# Add special tokens for unknown words and end of text
all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token:integer for integer, token in enumerate(all_tokens)}
vocab_size = len(all_tokens)
print(f"Vocabulary size (number unique tokens): {vocab_size}")

# Use improved tokenizer that recognizes text separation and unknown words
text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))
# print(text)

tokenizer = SimpleTokenizerV2(vocab)
# print(tokenizer.decode(tokenizer.encode(text)))

# Use Byte Pair Encoding (BPE) tokenizer
tokenizer = tiktoken.get_encoding("gpt2")
text = ("Hello, do you like tea? <|endoftext|> In the sunlit terraces of the someunknownPlace.")
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
# print(integers)
strings = tokenizer.decode(integers)
# print(strings)

############ Data sampling with sliding windows ################################################
# Encode the text using the BPE tokenizer
enc_text = tokenizer.encode(raw_text)
print(f"Length of the encoded text: {len(enc_text)}")

# Remove the first 50 tokens, just for illustration purposes
enc_sample = enc_text[50:]

# Create the windows
context_size = 4
x = enc_sample[:context_size]
y = enc_sample[1:context_size + 1]
print(f"x: {x}")
print(f"y: {y}")

for i in range(1, context_size + 1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))

# Using PyTorch
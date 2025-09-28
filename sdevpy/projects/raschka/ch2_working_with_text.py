import re
from importlib.metadata import version
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader
from sdevpy.llms.tokenizers import SimpleTokenizerV1, SimpleTokenizerV2
from sdevpy.projects.raschka import raschka_datasetloader as tdsl

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

############ Embedding ################################################
# Check out a few embeddings
vocab_size = 6
output_dim = 3
torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
print(embedding_layer.weight)

# Apply embeddings to token ID
print(embedding_layer(torch.tensor([3])))

# Apply to multiple IDs
input_ids = torch.tensor([2, 3, 5, 1])
print(embedding_layer(input_ids))

############ Encoding word position ################################################
vocab_size = 50257 # Size of BPE tokenizer
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

max_length = 4
dataloader = tdsl.create_dataloader_v1(raw_text, batch_size=8, max_length=max_length,
                                       stride=max_length, shuffle=False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Token IDs:\n", inputs)
print("\nInput shape: \n", inputs.shape)

token_embeddings = token_embedding_layer(inputs)
print("\nToken embedding shape: \n", token_embeddings.shape)

# Encode position (absolute)
context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
positions = torch.arange(context_length)
print("\nPositions: \n", positions)
pos_embeddings = pos_embedding_layer(positions)
print("\nPosition embedding shape: \n", pos_embeddings.shape)

# Add position to embedding
input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)


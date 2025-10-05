import torch


def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None,
             eos_id=None):
    for _ in range(max_new_tokens):
        # print(idx.shape)
        idx_cond = idx[:, -context_size:]
        # print(idx_cond.shape)
        with torch.no_grad():
            logits = model(idx_cond)

        # print(logits.shape)
        logits = logits[:, -1, :]

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float('-inf')).to(logits.device),
                                 logits)

        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if idx_next == eos_id:
            break

        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx contains a batch of sequences of token IDs
    for _ in range(max_new_tokens):
        # Pick the last context_size tokens in the sequence
        # If the sequence is smaller than context_size, then we take them all
        idx_cond = idx[:, -context_size:]

        # Evaluate the model on all sequences in the batch.
        # For each sequence in the batch, this responds a new sequence of equal number of tokens,
        # but where each element has an additional dimension. That additional dimention is the
        # 'logits', i.e. it has the size of the vocabulary and will represent (after softmax)
        # the probability of each token ID in the vocabulary.
        with torch.no_grad():
            logits = model(idx_cond)

        # For each element in the batch, select the last token in the sequence
        logits = logits[:, -1, :]

        # In the logit dimension, apply softmax to transform the numbers into probabilities
        probas = torch.softmax(logits, dim=-1)

        # Find the index of the max element along the probability dimension. That's the token ID
        # for the token that has the max probability
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)

        # Append that new token ID to the sequence of tokens, then iterate
        idx = torch.cat((idx, idx_next), dim=1)

    return idx

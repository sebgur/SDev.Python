import torch

# ToDo: would be interesting to understand, after training, what are the actual words
# in the rest of the context output by the model. That is, we only use the last element
# in the sequence and interpret it as the predicted word, but in reality the model
# outputs a whole sequence, out of which we only use the last token. But what do the
# other ones represent? Probably not the previous words in the sentence, as they have
# all been jumbled up.

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

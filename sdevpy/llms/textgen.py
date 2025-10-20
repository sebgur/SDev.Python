import torch


def format_answer(start_text, answer_text):
    answer = answer_text.replace(start_text, '') # Remove input
    answer = answer.lstrip()
    answer = answer.lstrip('\r\n')
    answer = answer.rstrip()
    answer = answer.rstrip('\r\n')
    return answer


def generate(model, idx, max_new_tokens, context_size, token_generator,
             eos_ids=None, max_sentences=4, eot_ids=None):
    n_sentences = 0
    for _ in range(max_new_tokens):
        idx_next = token_generator.next(model, idx, context_size)

        # print(idx_next)
        if eot_ids is not None:
            if idx_next in eot_ids:
                break

        idx = torch.cat((idx, idx_next), dim=1)

        if eos_ids is not None:
            if idx_next in eos_ids:
                n_sentences += 1
                if n_sentences >= max_sentences:
                    break

    return idx


# def generate_simple(model, idx, max_new_tokens, context_size, token_generator):
#     """ generate_text_simple in Raschka """
#     # idx contains a batch of sequences of token IDs
#     for _ in range(max_new_tokens):
#         idx_next = token_generator.next(model, idx, context_size)
#         # # Pick the last context_size tokens in the sequence
#         # # If the sequence is smaller than context_size, then we take them all
#         # idx_cond = idx[:, -context_size:]

#         # # Evaluate the model on all sequences in the batch.
#         # # For each sequence in the batch, this responds a new sequence of equal number of tokens,
#         # # but where each element has an additional dimension. That additional dimention is the
#         # # 'logits', i.e. it has the size of the vocabulary and will represent (after softmax)
#         # # the probability of each token ID in the vocabulary.
#         # with torch.no_grad():
#         #     logits = model(idx_cond)

#         # # For each element in the batch, select the last token in the sequence
#         # logits = logits[:, -1, :]

#         # # In the logit dimension, apply softmax to transform the numbers into probabilities
#         # probas = torch.softmax(logits, dim=-1)

#         # # Find the index of the max element along the probability dimension. That's the token ID
#         # # for the token that has the max probability
#         # idx_next = torch.argmax(probas, dim=-1, keepdim=True)

#         # Append that new token ID to the sequence of tokens, then iterate
#         idx = torch.cat((idx, idx_next), dim=1)

#     return idx


class NextTokenGenerator:
    """ Default inputs correspond to most simple text generation where the next token is
        most probable one """
    def __init__(self, top_k=None, temperature=0.0):
        self.top_k = top_k
        self.temperature = temperature

    def next(self, model, idx, context_size):
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

        if self.top_k is not None:
            top_logits, _ = torch.topk(logits, self.top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float('-inf')).to(logits.device),
                                 logits)

        if self.temperature > 0.0:
            logits = logits / self.temperature
            # In the logit dimension, apply softmax to transform the numbers into probabilities
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            # Find the index of the max element along the probability dimension. That's the token ID
            # for the token that has the max probability
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        return idx_next


# class SimpleTokenGenerator:
#     def next(self, model, idx, context_size):
#         idx_cond = idx[:, -context_size:]

#         with torch.no_grad():
#             logits = model(idx_cond)

#         logits = logits[:, -1, :]

#         probas = torch.softmax(logits, dim=-1)

#         return torch.argmax(probas, dim=-1, keepdim=True)


def text_to_tokens(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


def tokens_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

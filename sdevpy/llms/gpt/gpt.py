import os
import json
import urllib.request
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
from sdevpy.llms.gpt.attention import MultiHeadAttention


############ TODO #################################################################################
# * Get a fresh GPT model to run from weights and config
# * Plug under local_model design to harmonize chat/instruction design
# * Cleanup training algorithm and try it on runpod
# * Refresh notebook to work on email/trade booking instructions
# * Write LORA routine to train on trade booking


class GPTModel(nn.Module):
    """ The model's forward method takes in a batch of sequences of token IDs and it outputs
        a batch of sequences with the same size, i.e. representing the same number of tokens.
        However, these tokens are now represented by a whole dimension of logits, rather than
        by a single token ID as when they entered the model.
        These logits can then be interpreted/post-processed (through e.g. softmas) as
        probabilities along the vocabulary direction.
    """
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        # Note: clarify what nn.Linear(n1, n2) is: n2 neurons with n1 incoming connections?
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        # print(f"Input: {in_idx.shape}")

        tok_embeds = self.tok_emb(in_idx)
        # print(f"Token Embedding: {tok_embeds.shape}")
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        # print(f"Position Embedding: {pos_embeds.shape}")
        x = tok_embeds + pos_embeds
        # print(f"Embedding: {x.shape}")

        x = self.drop_emb(x) # Note: what is this doing, again?
        # print(f"After drop-out: {x.shape}")
        x = self.trf_blocks(x)
        # print(f"After Transformers: {x.shape}")
        x = self.final_norm(x)
        # print(f"After Final Normalization: {x.shape}")
        logits = self.out_head(x)
        # print(f"Output: {logits.shape}")

        return logits


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(d_in=cfg["emb_dim"], d_out=cfg["emb_dim"],
                                      context_length=cfg["context_length"],
                                      num_heads=cfg["n_heads"], dropout=cfg["drop_rate"],
                                      qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x # Shortcut for attention block
        # print("Transformer(before norm1): ", x.shape)
        x = self.norm1(x)
        # print("Transformer(before attention): ", x.shape)
        x = self.att(x)
        # print("Transformer(after attention): ", x.shape)
        x = self.drop_shortcut(x)
        x = x + shortcut # Add the original input back

        shortcut = x # Shortcut for feed-forward block
        # print("Transformer(before norm2): ", x.shape)
        x = self.norm2(x)
        # print("Transformer(before MLP): ", x.shape)
        x = self.ff(x)
        # print("Transformer(after MLP): ", x.shape)
        x = self.drop_shortcut(x)
        x = x + shortcut # Add the original input back

        return x


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
                                    GELU(),
                                    nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),)

    def forward(self, x):
        return self.layers(x)


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) *
                                         (x + 0.044715 * torch.pow(x, 3))))


class LayerNorm(nn.Module):
    """ Normalize the input along its embedding direction, then scale it using trained scale and shift
        parameters. """
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")

    return torch.nn.Parameter(torch.tensor(right))


def load_weights(gpt, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])

    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split((params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(gpt.trf_blocks[b].att.W_value.weight, v_w.T)
        q_b, k_b, v_b = np.split((params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(gpt.trf_blocks[b].att.W_value.bias, v_b)
        gpt.trf_blocks[b].att.out_proj.weight = assign(gpt.trf_blocks[b].att.out_proj.weight,
                                                       params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(gpt.trf_blocks[b].att.out_proj.bias,
                                                     params["blocks"][b]["attn"]["c_proj"]["b"])
        gpt.trf_blocks[b].ff.layers[0].weight = assign(gpt.trf_blocks[b].ff.layers[0].weight,
                                                       params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(gpt.trf_blocks[b].ff.layers[0].bias,
                                                     params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(gpt.trf_blocks[b].ff.layers[2].weight,
                                                       params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(gpt.trf_blocks[b].ff.layers[2].bias,
                                                     params["blocks"][b]["mlp"]["c_proj"]["b"])
        gpt.trf_blocks[b].norm1.scale = assign(gpt.trf_blocks[b].norm1.scale, params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(gpt.trf_blocks[b].norm1.shift, params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(gpt.trf_blocks[b].norm2.scale, params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(gpt.trf_blocks[b].norm2.shift, params["blocks"][b]["ln_2"]["b"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])

    # This looks like the so-called 'weight tying'
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])


if __name__ == "__main__":
    print("Hello")

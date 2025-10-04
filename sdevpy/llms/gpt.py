import torch
import torch.nn as nn
from sdevpy.llms.attention import MultiHeadAttention


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
        # ToDo: clarify what nn.Linear(n1, n2) is: n2 neurons with n1 incoming connections?
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        print(f"Input: {in_idx.shape}")

        tok_embeds = self.tok_emb(in_idx)
        print(f"Token Embedding: {tok_embeds.shape}")
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        print(f"Position Embedding: {pos_embeds.shape}")
        x = tok_embeds + pos_embeds
        print(f"Embedding: {x.shape}")

        x = self.drop_emb(x) # ToDo: what is this doing, again?
        print(f"After drop-out: {x.shape}")
        x = self.trf_blocks(x)
        print(f"After Transformers: {x.shape}")
        x = self.final_norm(x)
        print(f"After Final Normalization: {x.shape}")
        logits = self.out_head(x)
        print(f"Output: {logits.shape}")

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
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut # Add the original input back

        shortcut = x # Shortcut for feed-forward block
        x = self.norm2(x)
        x = self.ff(x)
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


class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(*[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))

        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

    def forward(self, x):
        return x


class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()

    def forward(self, x):
        return x

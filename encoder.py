import torch
from multihead import MultiHeadAttention
from ff import FF


class Encode(torch.nn.Module):
    def __init__(self, num_heads, n_embd, max_seq_length):
        super().__init__()
        self.ff = FF(n_embd)
        self.attn = MultiHeadAttention(num_heads, n_embd, max_seq_length)
        self.l1 = torch.nn.LayerNorm(n_embd)
        self.l2 = torch.nn.LayerNorm(n_embd)
        self.dropout1 = torch.nn.Dropout(0.2)
        self.dropout2 = torch.nn.Dropout(0.2)
        self.dropout3 = torch.nn.Dropout(0.2)

    def forward(self, x, mask=None):
        attn_out = self.attn(x, x, x, mask)
        x = self.l1(self.dropout1(attn_out) + x)
        ff_out = self.ff(x)
        ff_out = self.l2(self.dropout2(ff_out) + x)
        attn_out = self.attn(ff_out, ff_out, ff_out, mask)
        return self.l2(self.dropout3(attn_out) + ff_out)


class Encoder(torch.nn.Module):
    def __init__(self, vocab_size, max_seq_length, num_heads, num_layers, n_embd):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, n_embd)
        self.pos_embedding = torch.nn.Embedding(max_seq_length, n_embd)
        self.layers = torch.nn.ModuleList(
            [Encode(num_heads, n_embd, max_seq_length) for i in range(num_layers)]
        )
        self.norm = torch.nn.LayerNorm(n_embd)
        self.pad_token_id = 0

    def forward(self, x):
        seq_length = x.shape[1]
        positions = (
            torch.arange(0, seq_length, device=x.device).unsqueeze(0).expand_as(x)
        )
        x1 = self.embedding(x) + self.pos_embedding(positions)
        mask = (x != self.pad_token_id).float()
        mask = mask.unsqueeze(1)
        for layer in self.layers:
            x1 = layer(x1, mask)
        return self.norm(x1)

import torch
from multihead import MultiHeadAttention
class Encode(torch.nn.Module):
    def __init__(self, num_heads, n_embd, max_seq_length):
        super().__init__()
        self.ff = FF()
        self.attn = MultiHeadAttention(num_heads, n_embd, max_seq_length)
        self.l1 = torch.nn.LayerNorm(n_embd)
        self.l2 = torch.nn.LayerNorm(n_embd)
        self.dropout1 = torch.nn.Dropout(0.1)
        self.dropout2 = torch.nn.Dropout(0.1)

    def forward(self, x):
        attn_out = self.attn(x, x, x)
        x = self.l1(self.dropout1(attn_out) + x)
        ff_out = self.ff(x)
        attn_out = self.attn(x, x, x)
        return self.l2(self.dropout2(attn_out) + x)


class Encoder(torch.nn.Module):
    def __init__(self, vocab_size, max_seq_length, num_heads, num_layers, n_embd):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, n_embd)
        self.pos_embedding = torch.nn.Embedding(max_seq_length, n_embd)
        self.layers = torch.nn.ModuleList(
            [Encode(num_heads, n_embd, max_seq_length) for i in range(num_layers)]
        )
        self.norm = torch.nn.LayerNorm(n_embd)

    def forward(self, x):
        seq_length = x.shape[1]
        positions = (
            torch.arange(0, seq_length, device=x.device).unsqueeze(0).expand_as(x)
        )
        x = self.embedding(x) + self.pos_embedding(positions)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

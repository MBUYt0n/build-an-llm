import torch
from multihead import MultiHeadAttention
from ff import FF


class Decode(torch.nn.Module):
    def __init__(self, num_heads, n_embd, max_seq_length):
        super().__init__()
        self.attn1 = MultiHeadAttention(num_heads, n_embd, max_seq_length)
        self.attn2 = MultiHeadAttention(num_heads, n_embd, max_seq_length)
        self.norm1 = torch.nn.LayerNorm(n_embd)
        self.norm2 = torch.nn.LayerNorm(n_embd)
        self.norm3 = torch.nn.LayerNorm(n_embd)
        self.ff = FF(n_embd)
        self.dropout1 = torch.nn.Dropout(0.2)
        self.dropout2 = torch.nn.Dropout(0.2)
        self.dropout3 = torch.nn.Dropout(0.2)

    def forward(self, x, enc):
        attn_out = self.attn1(x, x, x)
        x = self.norm1(x + self.dropout1(attn_out))
        attn_out = self.attn2(x, enc, enc)
        x = self.norm2(x + self.dropout2(attn_out))
        return self.norm3(x + self.dropout3(self.ff(x)))


class Decoder(torch.nn.Module):
    def __init__(
        self,
        vocab_size,
        max_seq_length,
        num_layers,
        num_heads,
        n_embd,
    ):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, n_embd)
        self.pos_embedding = torch.nn.Embedding(max_seq_length, n_embd)
        self.layers = torch.nn.ModuleList(
            [Decode(num_heads, n_embd, max_seq_length) for i in range(num_layers)]
        )
        self.norm = torch.nn.LayerNorm(n_embd)

    def forward(self, x, enc_output):
        seq_length = x.size(1)
        positions = (
            torch.arange(0, seq_length, device=x.device).unsqueeze(0).expand_as(x)
        )
        x1 = self.embedding(x) + self.pos_embedding(positions)

        for layer in self.layers:
            x1 = layer(x1, enc_output)
        return self.norm(x1)

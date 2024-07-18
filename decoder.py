import torch
from multihead import MultiHeadAttention
from ff import FF


class Decode(torch.nn.Module):
    def __init__(self, num_heads, n_embd, max_seq_length):
        super().__init__()
        self.attn1 = MultiHeadAttention(num_heads, n_embd, max_seq_length)
        self.attn2 = MultiHeadAttention(num_heads)
        self.norm1 = torch.nn.LayerNorm(n_embd)
        self.norm2 = torch.nn.LayerNorm(n_embd)
        self.norm3 = torch.nn.LayerNorm(n_embd)
        self.ff = FF()

    def forward(self, x, enc):
        attn_out = self.attn1(x, x, x, 1)
        x = self.norm1(x + attn_out)
        attn_out = self.attn2(x, enc, enc, 1)
        x = self.norm2(x + attn_out)
        return self.norm3(x + self.ff(x))


class Decoder(torch.nn.Module):
    def __init__(
        self,
        vocab_size,
        max_seq_len,
        num_layers,
        num_heads,
        n_embd,
        hidden_dim,
        max_seq_length,
    ):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, n_embd)
        self.pos_embedding = torch.nn.Embedding(max_seq_len, n_embd)
        self.lstm = torch.nn.LSTM(
            n_embd, hidden_dim, batch_first=True
        )  # Initialize LSTM
        self.layers = torch.nn.ModuleList(
            [Decode(num_heads, n_embd, max_seq_length) for i in range(num_layers)]
        )
        self.norm = torch.nn.LayerNorm(n_embd)

    def forward(self, x, enc_output):
        seq_length = x.size(1)
        positions = (
            torch.arange(0, seq_length, device=x.device).unsqueeze(0).expand_as(x)
        )
        x = self.embedding(x) + self.pos_embedding(positions)
        x, _ = self.lstm(x)
        for layer in self.layers:
            x = layer(x, enc_output)
        return self.norm(x)

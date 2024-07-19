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

    def forward(self, x, enc, mask=None):
        attn_out = self.attn1(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_out))
        attn_out = self.attn2(x, enc, enc, 1)
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
        hidden_dim,
    ):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, n_embd)
        self.pos_embedding = torch.nn.Embedding(max_seq_length, n_embd)
        self.lstm = torch.nn.LSTM(
            n_embd, hidden_dim, batch_first=True
        )  
        self.layers = torch.nn.ModuleList(
            [Decode(num_heads, n_embd, max_seq_length) for i in range(num_layers)]
        )
        self.norm = torch.nn.LayerNorm(n_embd)
        self.pad_token_id = 0

    def forward(self, x, enc_output):
        seq_length = x.size(1)
        positions = (
            torch.arange(0, seq_length, device=x.device).unsqueeze(0).expand_as(x)
        )
        x = self.embedding(x) + self.pos_embedding(positions)
        x, _ = self.lstm(x)

        mask = (x != self.pad_token_id).unsqueeze(1).unsqueeze(2).float()
        mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

        for layer in self.layers:
            x = layer(x, enc_output, mask)
        return self.norm(x)

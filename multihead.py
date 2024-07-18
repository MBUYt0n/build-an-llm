import torch
from head import Head

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, num_heads, n_embd, max_seq_length):
        super().__init__()
        self.heads = torch.nn.ModuleList(
            [
                Head(n_embd, n_embd // num_heads, max_seq_length)
                for i in range(num_heads)
            ]
        )
        self.out = torch.nn.Linear(n_embd, n_embd)

    def forward(self, q, k, v, mask=None):
        head_out = [head(q, k, v, mask) for head in self.heads]
        concat = torch.cat(head_out, dim=-1)
        return self.out(concat)

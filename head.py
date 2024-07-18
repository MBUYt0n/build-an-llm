import torch


class Head(torch.nn.Module):
    def __init__(self, n_embd, head_size, max_seq_length):
        super().__init__()
        self.head_size = head_size
        self.key = torch.nn.Linear(n_embd, self.head_size, bias=False)
        self.query = torch.nn.Linear(n_embd, self.head_size, bias=False)
        self.values = torch.nn.Linear(n_embd, self.head_size, bias=False)
        self.scale_factor = self.head_size**-0.5
        self.max_seq_length = max_seq_length

    def forward(self, q, k, v, mask=None):
        k = self.key(k)
        q = self.query(q)
        v = self.values(v)
        w = (q @ k.transpose(-2, -1)) * self.scale_factor
        if mask:
            device = w.device
            tril = torch.tril(
                torch.ones(self.max_seq_length, self.max_seq_length, device=device)
            )
            seq_length = q.size(1)  # Get the sequence length from q
            w = w.masked_fill(tril[:seq_length, :seq_length] == 0, float("-inf"))
        w = torch.nn.functional.softmax(w, dim=-1)
        return w @ v

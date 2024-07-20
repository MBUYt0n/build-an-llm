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

        if mask is not None:
            w = w.masked_fill(mask == 0, float("1e-9"))
        w = torch.nn.functional.softmax(w, dim=-1)
        return w @ v

import torch


class FF(torch.nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.linear1 = torch.nn.Linear(n_embd, 8 * n_embd)
        self.linear2 = torch.nn.Linear(8 * n_embd, n_embd)

    def forward(self, x):
        return self.linear2(torch.nn.functional.relu(self.linear1(x)))

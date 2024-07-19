import torch


class tokenizer:
    def __init__(self, x):
        self.x = x

    def fit(self):
        tokens = set("".join(self.x))
        self.vocab_size = len(tokens) + 1
        self.tokens = {i: j for i, j in zip(tokens, range(1, self.vocab_size))}
        self.tokens["<PAD>"] = 0
        self.m = max([len(i) for i in self.x])
        self.detoken = {j: i for i, j in self.tokens.items()}

    def encode(self, x):
        inputs = torch.zeros((len(x), self.m), dtype=torch.int64)
        for i in range(len(x)):
            for j in range(len(x[i])):
                inputs[i, j] = self.tokens[x[i][j]]

        return inputs

    def decode(self, x):
        return "".join([self.detoken[int(i)] for i in x if i != 0])

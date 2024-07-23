import torch
import re


class tokenizer:
    def __init__(self, x):
        self.x = x

    def preprocess(self, x=None):
        x = x if x else self.x
        o = []
        for i in x:
            l = []
            for j in i:
                j = re.sub(r"[^a-zA-z0-9]", "", j.lower())
                if j:
                    l.append(j)
            o.append(l)
        return o

    def fit(self):
        self.x = self.preprocess()
        tokens = set([j for i in self.x for j in i])
        self.vocab_size = len(tokens) + 1
        self.tokens = {i: j for i, j in zip(tokens, range(1, self.vocab_size))}
        self.tokens["<PAD>"] = 0
        self.m = max([len(i) for i in self.x])
        self.detoken = {j: i for i, j in self.tokens.items()}

    def encode(self, x):
        x = self.preprocess(x)
        inputs = torch.zeros((len(x), self.m), dtype=torch.int64)
        for i in range(len(x)):
            for j in range(len(x[i])):
                try:
                    inputs[i, j] = self.tokens[x[i][j]]
                except:
                    inputs[i, j] = 0

        return inputs

    def decode(self, x):
        return " ".join([self.detoken[int(i)] for i in x if i != 0])

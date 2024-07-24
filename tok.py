import torch
import re
import gc


class tokenizer:
    def __init__(self, x):
        self.x = x

    def preprocess(self, x):
        o = []
        x = x.split()
        for i in x:
            j = re.sub(r"[^a-zA-z0-9]", "", i.lower())
            o.append(j)
        return o

    def fit(self, x=None):
        if x is None:
            x = self.x
        tokens = set()
        for i in x:
            tokens = tokens.union(set(self.preprocess(i)))
        self.vocab_size = len(tokens) + 1
        self.tokens = {i: j for i, j in zip(tokens, range(1, self.vocab_size))}
        self.tokens["<PAD>"] = 0
        self.detoken = {j: i for i, j in self.tokens.items()}
        del self.x
        gc.collect()

    def encode(self, x):
        l = []
        for i in x:
            l.append(self.preprocess(i))
        del x
        gc.collect()

        inputs = []
        for i in range(len(l)):
            inputs.append([])
            for j in range(len(l[i])):
                try:
                    inputs[i].append(self.tokens[l[i][j]])
                except:
                    inputs[i].append(0)

        return inputs

    def decode(self, x):
        return " ".join([self.detoken[int(i)] for i in x if i != 0])

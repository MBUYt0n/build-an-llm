from transformers import GPT2Tokenizer
import torch
import os
import numpy as np

class Tokenize:
    def __init__(self, corpus):
        self.corpus = corpus
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    def tokenize(self):
        tok = []
        toks = []
        for i in self.corpus:
            tok.append(self.tokenizer.encode(i, return_tensors="pt")[0])
            toks.append(torch.cat(tok, dim=0))
        return toks

    def get_data(self, seq_length):
        inps = []
        toks = self.tokenize()
        for j in toks:
            for i in range(seq_length, len(j) - 1):
                inps.append(j[i - seq_length : i + 1])
        return torch.utils.data.DataLoader(inps, batch_size=32, shuffle=True)

from transformers import BertTokenizer
import torch


class Tokenize:
    def __init__(self, corpus):
        self.corpus = corpus
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def tokenize(self):
        toks = []
        for i in self.corpus:
            toks.append(self.tokenizer.encode(i, return_tensors="pt")[0])
        return toks

    def get_data(self, seq_length):
        inps = []
        toks = self.tokenize()
        for j in toks:
            for i in range(seq_length, len(j) - 1):
                inps.append(j[i - seq_length: i + 1])
        return torch.utils.data.DataLoader(inps, batch_size=32, shuffle=True)

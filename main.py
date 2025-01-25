import torch
from tok import Tokenize
from llm import LLM
import numpy as np
import os
from train_gpu import Trainer


l = os.listdir("data")
corpus = []
for i in l:
    with open("data/" + i, "r") as f:
        corpus.append(f.read())
        f.close()
t = Tokenize(corpus)
data = t.get_data(10)
a = next(iter(data))
enc = e(a[:, :-1])
print(d(a[:, :-1], enc).shape)

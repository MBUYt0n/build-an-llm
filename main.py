import torch
from tok import Tokenize
from llm import LLM
import numpy as np
import os
from train_gpu import Trainer


l = os.listdir("data")[:2]
k = os.listdir("data")[3:4]
valcorpus = []
for i in k:
    with open("data/" + i, "r") as f:
        valcorpus.append(f.read())
        f.close()
valt = Tokenize(valcorpus)
valdata = valt.get_data(10)

corpus = []
for i in l:
    with open("data/" + i, "r") as f:
        corpus.append(f.read())
        f.close()
t = Tokenize(corpus)
data = t.get_data(10)
model = LLM(50257, 100, 4, 4, 768).to("cuda")
t = Trainer(model)
t.train(data, 10)

a = next(iter(valdata))[0]
a = a.to("cuda")
x = model.generate(a.unsqueeze(0), 50)
print(valt.decode(x))

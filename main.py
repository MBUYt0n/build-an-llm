import torch
from tok import tokenizer
from llm import llm
import numpy as np
import os
from train_gpu import Trainer

l = os.listdir("data")
x = []
for i in l:
    f = open(
        f"data/{i}",
        "r",
        errors="replace",
    )
    x.append(f.read())


tok = tokenizer(x)
tok.fit()
inputs = []
inputs = tok.encode(x)
vocab_size = tok.vocab_size

# a = tok.encode(["Black widow was"])
# print(a)
batch_size = 32
seq_length = 256
max_seq_length = 256
n_embd = 256

evals = inputs[-2:]
inputs = inputs[:-2]

device = torch.device("cuda")
model = llm(
    vocab_size, max_seq_length=max_seq_length, num_heads=8, num_layers=4, n_embd=n_embd
).to(device)
t = Trainer(model)
s, _ = t.batches(inputs, 1, 256)

# t.train(inputs, evals, batch_size, seq_length)

# torch.save(model.state_dict(), "model.pth")
# model.load_state_dict(torch.load("model.pth"))

# a = tok.encode(["Black widow was".split()])
# s, _ = t.create_batches(a, 1, 256)
# s = s[0].to(device)
# om = model.generate(input_ids=s, max_length=300)
# print(om.shape)
# om = om.reshape(-1)
# print(om.shape)
# print(tok.decode(om))

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
    x.append(f.read().split())


tok = tokenizer(x)
tok.fit()
inputs = tok.encode(x)
vocab_size = tok.vocab_size

batch_size = 32
seq_length = 256
max_seq_length = 256
n_embd = 256

# evals = inputs[-2:]
# inputs = inputs[:-2]

device = torch.device("cpu")
model = llm(
    vocab_size, max_seq_length=max_seq_length, num_heads=8, num_layers=4, n_embd=n_embd
).to(device)
t = Trainer(model)
# t.train(inputs, evals, batch_size, seq_length)

# torch.save(model.state_dict(), "model.pth")
model.load_state_dict(torch.load("model.pth"))

a = tok.encode("Black widow was".split())
s, _ = t.create_batches(a, 1, 256)
d = []
for i in s:
    d.append(i[0])
print(d)
s = torch.stack(d).to(device)
om = model.generate(s)
print(tok.decode(om[0]))


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

evals = inputs[-2:]
inputs = inputs[:-2]

device = torch.device("cuda")
model = llm(
    vocab_size, max_seq_length=max_seq_length, num_heads=8, num_layers=4, n_embd=n_embd
).to(device)

t = Trainer(model)
t.train(inputs, evals, batch_size, seq_length)

torch.save(model.state_dict(), "model.pth")

a = torch.randint(0, vocab_size, (1, 256)).to(device)
om = model.generate(a)
print(tok.decode(om[0]))

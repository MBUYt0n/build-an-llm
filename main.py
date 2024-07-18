import torch
from tok import tokenizer
from llm import llm
import numpy as np
import os
from train_cpu import Trainer

def create_batches(input_data, batch_size, seq_length):
    num_samples, total_length = input_data.shape
    num_chunks = total_length // seq_length + (total_length % seq_length != 0)

    chunks = []
    out_chunks = []
    for i in range(num_samples):
        for j in range(num_chunks):
            start_idx = j * seq_length
            end_idx = min(start_idx + seq_length, total_length)
            chunk = input_data[i, start_idx:end_idx]

            out_start_idx = start_idx + 1
            out_end_idx = min(out_start_idx + seq_length, total_length)
            out_chunk = input_data[i, out_start_idx:out_end_idx]

            if end_idx - start_idx < seq_length:
                padding = torch.zeros(
                    seq_length - (end_idx - start_idx), dtype=chunk.dtype
                )
                chunk = torch.cat([chunk, padding])
                out_padding = torch.zeros(
                    seq_length - (out_end_idx - out_start_idx), dtype=out_chunk.dtype
                )
                out_chunk = torch.cat([out_chunk, out_padding])

            chunks.append(chunk)
            out_chunks.append(out_chunk)

    chunks = torch.stack(chunks)
    num_batches = chunks.size(0) // batch_size
    batches = torch.split(chunks, batch_size)

    out_chunks = torch.stack(out_chunks)
    num_out_batches = out_chunks.size(0) // batch_size
    out_batches = torch.split(out_chunks, batch_size)

    return batches, out_batches


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
inputs = tok.fit()
vocab_size = tok.vocab_size

batch_size = 64
seq_length = 256
max_seq_length = 256
n_embd = 256

device = torch.device("cpu")
model = llm(
    vocab_size, max_seq_length=max_seq_length, num_heads=4, num_layers=2, n_embd=n_embd
).to(device)

t = Trainer(create_batches, model)
t.train(inputs, batch_size, seq_length)
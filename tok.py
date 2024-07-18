import numpy as np
import os

l = os.listdir("/kaggle/input/marvel-cinematic-universe-dialogue-dataset")
x = []
for i in l:
    f = open(
        f"/kaggle/input/marvel-cinematic-universe-dialogue-dataset/{i}",
        "r",
        errors="replace",
    )
    x.append(f.read())

m = 0
for i in x:
    m = m if len(i) < m else len(i)
m


tokens = set("".join(x))
vocab_size = len(tokens)
vocab_size


tokens = {i: j for i, j in zip(tokens, range(vocab_size))}
inputs = []
for i in x:
    inputs.append([])
    for j in i:
        inputs[-1].append(tokens[j])
len(inputs)

from tensorflow.keras.preprocessing.sequence import pad_sequences

inputs = pad_sequences(inputs, maxlen=m)

print(inputs.shape)

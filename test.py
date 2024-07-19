# import torch
# import os
# from tok import tokenizer


# l = os.listdir("data")
# x = []
# for i in l:
#     f = open(
#         f"data/{i}",
#         "r",
#         errors="replace",
#     )
#     x.append(f.read())

# tok = tokenizer(x)
# tok.fit()
# x = torch.tensor([[1, 2, 3, 0, 0]])  # Example input with padding
# mask = (x != tok.tokens["<PAD>"]).unsqueeze(1).unsqueeze(2).float()
# mask = mask.masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
# print(mask)

import torch

a = torch.randint(0, 10, (16, 1, 1, 256, 256))
a = a[:, 0, 0, :, :]
print(a.shape)
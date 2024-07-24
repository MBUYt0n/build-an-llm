# # import torch
# # import os
# # from tok import tokenizer


# # l = os.listdir("data")
# # x = []
# # for i in l:
# #     f = open(
# #         f"data/{i}",
# #         "r",
# #         errors="replace",
# #     )
# #     x.append(f.read())

# # tok = tokenizer(x)
# # tok.fit()
# # x = torch.tensor([[1, 2, 3, 0, 0]])  # Example input with padding
# # mask = (x != tok.tokens["<PAD>"]).unsqueeze(1).unsqueeze(2).float()
# # mask = mask.masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
# # print(mask)

# # import torch

# # a = torch.randint(0, 10, (16, 1, 1, 256, 256))
# # a = a[:, 0, 0, :, :]
# # print(a.shape)

# # import torch

# # # Example input tensor (batch_size, seq_length)
# # input_tensor = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 6, 7, 0]])  # 0 is the pad token

# # # pad_token_id is 0
# # pad_token_id = 0

# # # Create the attention mask (1 for non-pad tokens, 0 for pad tokens)
# # attention_mask = (input_tensor != pad_token_id).float()
# # print(attention_mask.shape)
# # attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
# # print(attention_mask.shape)


# import torch


# class Head(torch.nn.Module):
#     def __init__(self, n_embd, head_size, max_seq_length):
#         super().__init__()
#         self.head_size = head_size
#         self.key = torch.nn.Linear(n_embd, self.head_size, bias=False)
#         self.query = torch.nn.Linear(n_embd, self.head_size, bias=False)
#         self.values = torch.nn.Linear(n_embd, self.head_size, bias=False)
#         self.scale_factor = self.head_size**-0.5
#         self.max_seq_length = max_seq_length

#     def forward(self, q, k, v, mask=None):
#         k = self.key(k)
#         q = self.query(q)
#         v = self.values(v)
#         w = (q @ k.transpose(-2, -1)) * self.scale_factor

#         if mask is not None:
#             print(f"Shape of w before masking: {w.shape}")
#             w = w.masked_fill(mask == 0, float("-inf"))
#             print(f"Shape of w after masking: {w.shape}")
#         w = torch.nn.functional.softmax(w, dim=-1)
#         return w @ v


# import torch
# import os


# class tokenizer:
#     def __init__(self, x):
#         self.x = x

#     def fit(self):
#         tokens = set("".join(self.x))
#         self.vocab_size = len(tokens) + 1
#         self.tokens = {i: j for i, j in zip(tokens, range(1, self.vocab_size))}
#         self.tokens["<PAD>"] = 0
#         self.m = max([len(i) for i in self.x])
#         self.detoken = {j: i for i, j in self.tokens.items()}

#     def encode(self, x):
#         inputs = torch.zeros((len(x), self.m), dtype=torch.int64)
#         for i in range(len(x)):
#             for j in range(len(x[i])):
#                 inputs[i, j] = self.tokens[x[i][j]]

#         return inputs

#     def decode(self, x):
#         return "".join([self.detoken[int(i)] for i in x if i != 0])


# def create_batches(input_data, batch_size, seq_length):
#     num_samples, total_length = input_data.shape
#     num_chunks = total_length // seq_length + (total_length % seq_length != 0)

#     chunks = []
#     out_chunks = []
#     for i in range(num_samples):
#         for j in range(num_chunks):
#             start_idx = j * seq_length
#             end_idx = min(start_idx + seq_length, total_length)
#             chunk = input_data[i, start_idx:end_idx]

#             out_start_idx = start_idx + 1
#             out_end_idx = min(out_start_idx + seq_length, total_length)
#             out_chunk = input_data[i, out_start_idx:out_end_idx]

#             if end_idx - start_idx < seq_length:
#                 padding = torch.zeros(
#                     seq_length - (end_idx - start_idx), dtype=chunk.dtype
#                 )
#                 chunk = torch.cat([chunk, padding])
#                 out_padding = torch.zeros(
#                     seq_length - (out_end_idx - out_start_idx), dtype=out_chunk.dtype
#                 )
#                 out_chunk = torch.cat([out_chunk, out_padding])

#             chunks.append(chunk)
#             out_chunks.append(out_chunk)

#     chunks = torch.stack(chunks)
#     num_batches = chunks.size(0) // batch_size
#     batches = torch.split(chunks, batch_size)

#     out_chunks = torch.stack(out_chunks)
#     num_out_batches = out_chunks.size(0) // batch_size
#     out_batches = torch.split(out_chunks, batch_size)

#     return batches, out_batches


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
# inputs = tok.encode(x)
# vocab_size = tok.vocab_size

# batch_size = 8
# seq_length = 256
# max_seq_length = 256
# n_embd = 256

# s, o = create_batches(inputs, batch_size, seq_length)
# a = s[0]

# head = Head(n_embd, 64, max_seq_length)
# mask = (a != 0).unsqueeze(1).unsqueeze(2).float()
# print(f"Shape of mask: {mask.shape}")
# out = head(a, a, a, mask)
# print(out.shape)

# import torch

# b = torch.randint(0, 10, (8, 256, 256)).float()

# for i in range(10):
#     a = torch.randint(0, 10, (8, 256, 256), dtype=torch.float32)
#     b = b.masked_fill(a == 0, float("-inf"))
#     print(b.shape)
# a = torch.nn.functional.softmax(b, dim=-1).shape
# print("A", a)

with open("/home/shusrith/projects/blind-eyes/build-an-llm/data/Captain.Marvel.txt", "r") as f:
    x = f.read().split()
    print(set(x))
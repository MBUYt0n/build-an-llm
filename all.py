import torch
from torch.utils.tensorboard import SummaryWriter
import gc


class Head(torch.nn.Module):
    def __init__(self, n_embd, head_size, max_seq_length):
        super().__init__()
        self.head_size = head_size
        self.key = torch.nn.Linear(n_embd, self.head_size, bias=False)
        self.query = torch.nn.Linear(n_embd, self.head_size, bias=False)
        self.values = torch.nn.Linear(n_embd, self.head_size, bias=False)
        self.scale_factor = self.head_size**-0.5
        self.max_seq_length = max_seq_length

    def forward(self, q, k, v, mask=None):
        k = self.key(k)
        q = self.query(q)
        v = self.values(v)
        w = (q @ k.transpose(-2, -1)) * self.scale_factor

        w = torch.nn.functional.softmax(w, dim=-1)
        w = torch.nn.functional.softmax(w, dim=-1)
        return w @ v


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, num_heads, n_embd, max_seq_length):
        super().__init__()
        self.heads = torch.nn.ModuleList(
            [
                Head(n_embd, n_embd // num_heads, max_seq_length)
                for i in range(num_heads)
            ]
        )
        self.out = torch.nn.Linear(n_embd, n_embd)

    def forward(self, q, k, v, mask=None):
        head_out = [head(q, k, v, mask) for head in self.heads]
        concat = torch.cat(head_out, dim=-1)
        return self.out(concat)


class FF(torch.nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.linear1 = torch.nn.Linear(n_embd, 4 * n_embd)
        self.linear2 = torch.nn.Linear(4 * n_embd, n_embd)

    def forward(self, x):
        return self.linear2(torch.nn.functional.relu(self.linear1(x)))


class Encode(torch.nn.Module):
    def __init__(self, num_heads, n_embd, max_seq_length):
        super().__init__()
        self.ff = FF(n_embd)
        self.attn = MultiHeadAttention(num_heads, n_embd, max_seq_length)
        self.l1 = torch.nn.LayerNorm(n_embd)
        self.l2 = torch.nn.LayerNorm(n_embd)
        self.dropout1 = torch.nn.Dropout(0.1)
        self.dropout2 = torch.nn.Dropout(0.1)

    def forward(self, x, mask=None):
        attn_out = self.attn(x, x, x)
        x = self.l1(self.dropout1(attn_out) + x)
        ff_out = self.ff(x)
        attn_out = self.attn(ff_out, ff_out, ff_out)
        return self.l2(self.dropout2(attn_out) + ff_out)


class Encoder(torch.nn.Module):
    def __init__(self, vocab_size, max_seq_length, num_heads, num_layers, n_embd):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, n_embd)
        self.pos_embedding = torch.nn.Embedding(max_seq_length, n_embd)
        self.layers = torch.nn.ModuleList(
            [Encode(num_heads, n_embd, max_seq_length) for i in range(num_layers)]
        )
        self.norm = torch.nn.LayerNorm(n_embd)
        self.pad_token_id = 0

    def forward(self, x):
        seq_length = x.shape[1]
        positions = (
            torch.arange(0, seq_length, device=x.device).unsqueeze(0).expand_as(x)
        )
        x = self.embedding(x) + self.pos_embedding(positions)


        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


import torch
from multihead import MultiHeadAttention
from ff import FF


class Decode(torch.nn.Module):
    def __init__(self, num_heads, n_embd, max_seq_length):
        super().__init__()
        self.attn1 = MultiHeadAttention(num_heads, n_embd, max_seq_length)
        self.attn2 = MultiHeadAttention(num_heads, n_embd, max_seq_length)
        self.norm1 = torch.nn.LayerNorm(n_embd)
        self.norm2 = torch.nn.LayerNorm(n_embd)
        self.norm3 = torch.nn.LayerNorm(n_embd)
        self.ff = FF(n_embd)

    def forward(self, x, enc, mask=None):
        attn_out = self.attn1(x, x, x, 1)
        x = self.norm1(x + attn_out)
        attn_out = self.attn2(x, enc, enc, 1)
        x = self.norm2(x + attn_out)
        return self.norm3(x + self.ff(x))


class Decoder(torch.nn.Module):
    def __init__(
        self,
        vocab_size,
        max_seq_length,
        num_layers,
        num_heads,
        n_embd,
        hidden_dim,
    ):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, n_embd)
        self.pos_embedding = torch.nn.Embedding(max_seq_length, n_embd)
        self.lstm = torch.nn.LSTM(n_embd, hidden_dim, batch_first=True)
        self.layers = torch.nn.ModuleList(
            [Decode(num_heads, n_embd, max_seq_length) for i in range(num_layers)]
        )
        self.norm = torch.nn.LayerNorm(n_embd)
        self.pad_token_id = 0

    def forward(self, x, enc_output):
        seq_length = x.size(1)
        positions = (
            torch.arange(0, seq_length, device=x.device).unsqueeze(0).expand_as(x)
        )
        x = self.embedding(x) + self.pos_embedding(positions)
        x, _ = self.lstm(x)

        mask = (x != self.pad_token_id).unsqueeze(1).unsqueeze(2).float()
        mask = mask.masked_fill(mask == 0, float("-inf")).masked_fill(
            mask == 1, float(0.0)
        )

        for layer in self.layers:
            x = layer(x, enc_output, mask)
        return self.norm(x)


class tokenizer:
    def __init__(self, x):
        self.x = x

    def fit(self):
        tokens = set("".join(self.x))
        self.vocab_size = len(tokens) + 1
        self.tokens = {i: j for i, j in zip(tokens, range(1, self.vocab_size))}
        self.tokens["<PAD>"] = 0
        self.m = max([len(i) for i in self.x])
        self.detoken = {j: i for i, j in self.tokens.items()}

    def encode(self, x):
        inputs = torch.zeros((len(x), self.m), dtype=torch.int64)
        for i in range(len(x)):
            for j in range(len(x[i])):
                inputs[i, j] = self.tokens[x[i][j]]

        return inputs

    def decode(self, x):
        return "".join([self.detoken[int(i)] for i in x if i != 0])


class llm(torch.nn.Module):
    def __init__(self, vocab_size, max_seq_length, num_heads, num_layers, n_embd):
        super().__init__()
        self.enc = Encoder(vocab_size, max_seq_length, num_heads, num_layers, n_embd)
        self.dec = Decoder(
            vocab_size, max_seq_length, num_heads, num_layers, n_embd, hidden_dim=n_embd
        )
        self.out = torch.nn.Linear(n_embd, vocab_size)
        self.max_seq_length = max_seq_length
        self.vocab_size = vocab_size

    def forward(self, x, y=None, enc_out=None):
        if enc_out is None:
            enc_out = self.enc(x)
        if y is not None:
            dec_out = self.dec(y, enc_out)
            return self.out(dec_out)
        return enc_out

    def generate(self, input_ids, max_length=50):
        self.eval()
        with torch.no_grad():
            enc_out = self.forward(input_ids)
            generated = input_ids

            for _ in range(max_length - input_ids.size(1)):
                output = self.forward(y=generated, enc_out=enc_out)
                next_token_logits = output[:, -1, :]
                # next_token_probs = torch.nn.functional.softmax(
                #     next_token_logits, dim=-1
                # )
                next_token_id = next_token_probs.argmax(dim=-1).unsqueeze(-1)
                generated = torch.cat([generated, next_token_id], dim=1)
        self.train()
        return generated


class Trainer:
    def __init__(self, func, model):
        self.batch = func
        self.model = model
        self.lossFn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)

    def train(self, inputs, batch_size, seq_length):
        self.model.train()
        writer = SummaryWriter()

        device = torch.device("cuda")

        # Number of epochs
        num_epochs = 10

        scaler = torch.cuda.amp.GradScaler()
        s, o = self.batch(inputs, batch_size, seq_length)
        for epoch in range(num_epochs):
            for i, (a, b) in enumerate(zip(s, o)):
                a = a.to(device)
                b = b.to(device)

                with torch.cuda.amp.autocast():
                    logits = self.model(a, b)
                    loss = self.lossFn(
                        logits.view(-1, self.model.vocab_size), b.view(-1).long()
                    )

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}], Loss: {loss.item()}"
                )

                writer.add_scalar("Loss/train", loss.item(), epoch * len(inputs) + i)

                a = a.cpu()
                b = b.cpu()
                logits = logits.cpu()
                loss = loss.cpu()
                del a, b, logits, loss
                torch.cuda.empty_cache()
                gc.collect()

        writer.close()


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
tok.fit()
inputs = tok.encode(x)
vocab_size = tok.vocab_size

batch_size = 32
seq_length = 256
max_seq_length = 256
n_embd = 256

device = torch.device("cuda")
model = llm(
    vocab_size, max_seq_length=max_seq_length, num_heads=4, num_layers=2, n_embd=n_embd
).to(device)

t = Trainer(create_batches, model)
t.train(inputs, batch_size, seq_length)

torch.save(model.state_dict(), "model.pth")

a = torch.randint(0, vocab_size, (1, 256)).to(device)
om = model.generate(a)
print(tok.decode(om[0]))

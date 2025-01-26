import torch
from encoder import Encoder
from decoder import Decoder


class LLM(torch.nn.Module):
    def __init__(self, vocab_size, max_seq_length, num_heads, num_layers, n_embd):
        super().__init__()
        self.enc = Encoder(vocab_size, max_seq_length, num_heads, num_layers, n_embd)
        self.dec = Decoder(vocab_size, max_seq_length, num_heads, num_layers, n_embd)
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
        output = [int(i) for i in input_ids[0]]
        with torch.no_grad():
            for _ in range(max_length):
                input_ids = input_ids.to("cuda")
                enc_out = self.forward(input_ids)
                generated = enc_out[:, -1, :].softmax(dim=-1).argmax(dim=-1)
                output.append(generated.item())
                input_ids = torch.cat([input_ids, generated.unsqueeze(0)], dim=1)

        return output


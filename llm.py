import torch
from encoder import Encoder
from decoder import Decoder


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
            enc_out = self.forward(
                input_ids
            )  # Assuming this runs the encoder and returns the output
            generated = input_ids

            for _ in range(max_length):
                output = self.forward(input_ids, generated, enc_out=enc_out)
                next_token_logits = output[:, -1, :]
                next_token_probs = torch.nn.functional.softmax(
                    next_token_logits, dim=-1
                )
                next_token_id = next_token_probs.argmax(dim=-1).unsqueeze(-1)
                generated = torch.cat([generated[:, 1:], next_token_id], dim=-1)
        return generated

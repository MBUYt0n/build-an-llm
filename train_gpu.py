import torch
import gc
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    def __init__(self, model):
        self.model = model
        self.lossFn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, "min", patience=2, factor=0.5
        )

    def create_batches(self, input_data, batch_size, seq_length):
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
                        seq_length - (out_end_idx - out_start_idx),
                        dtype=out_chunk.dtype,
                    )
                    out_chunk = torch.cat([out_chunk, out_padding])

                chunks.append(chunk)
                out_chunks.append(out_chunk)

        chunks = torch.stack(chunks)
        batches = torch.split(chunks, batch_size)

        out_chunks = torch.stack(out_chunks)
        out_batches = torch.split(out_chunks, batch_size)

        return batches, out_batches

    def train(self, inputs, evals, batch_size, seq_length, num_epochs=25):
        self.model.train()
        writer = SummaryWriter()

        device = torch.device("cuda")
        s, o = self.create_batches(inputs, batch_size, seq_length)
        scaler = torch.cuda.amp.GradScaler()

        for epoch in range(num_epochs):
            epoch_loss = 0

            for i, (a, b) in enumerate(zip(s, o)):
                a = a.to(device)
                b = b.to(device)

                with torch.cuda.amp.autocast():
                    logits = self.model(a, b)
                    logits = logits.view(-1, self.model.vocab_size)
                    b = b.view(-1).long()
                    loss = self.lossFn(logits, b)

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                epoch_loss += loss.item()

                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}], Loss: {loss.item()}"
                )

                writer.add_scalar("Loss/train", loss.item(), epoch * len(s) + i)

                a = a.cpu()
                b = b.cpu()
                logits = logits.cpu()
                loss = loss.cpu()
                del a, b, logits, loss
                torch.cuda.empty_cache()
                gc.collect()

            avg_epoch_loss = epoch_loss / len(s)
            print(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_epoch_loss}")
            writer.add_scalar("Loss/epoch", avg_epoch_loss, epoch)

            self.evaluate(evals, batch_size, seq_length, device)
            self.scheduler.step(epoch_loss)

        writer.close()

    def evaluate(self, inputs, batch_size, seq_length, device):
        self.model.eval()
        val_loss = 0

        s, o = self.create_batches(inputs, batch_size, seq_length)
        with torch.no_grad():
            for i, (a, b) in enumerate(zip(s, o)):
                a = a.to(device)
                b = b.to(device)

                logits = self.model(a, b)
                loss = self.lossFn(
                    logits.view(-1, self.model.vocab_size), b.view(-1).long()
                )
                val_loss += loss.item()

                a = a.cpu()
                b = b.cpu()
                logits = logits.cpu()
                loss = loss.cpu()
                del a, b, logits, loss
                torch.cuda.empty_cache()
                gc.collect()

        val_loss /= len(s)
        print(f"Validation Loss: {val_loss}")
        self.model.train()

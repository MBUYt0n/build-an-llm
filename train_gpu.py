import torch
import gc
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    def __init__(self, func, model):
        self.batch = func
        self.model = model
        self.lossFn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=2, factor=0.5)

    def train(self, inputs, batch_size, seq_length, num_epochs=10):
        self.model.train()
        writer = SummaryWriter()

        device = torch.device("cuda")
        s, o = self.batch(inputs, batch_size, seq_length)
        scaler = torch.cuda.amp.GradScaler()

        for epoch in range(num_epochs):
            epoch_loss = 0

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

            self.evaluate(inputs, batch_size, seq_length, device)
            self.scheduler.step(epoch_loss)

        writer.close()

    def evaluate(self, inputs, batch_size, seq_length, device):
        self.model.eval()
        val_loss = 0

        s, o = self.batch(inputs, batch_size, seq_length)
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

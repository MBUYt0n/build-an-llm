import torch
from torch.utils.tensorboard import SummaryWriter
import gc


class Trainer():
    def __init__(self, func, model):
        self.batch = func
        self.model = model
        self.lossFn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)


    def train(self, inputs, batch_size, seq_length):
        self.model.train()
        writer = SummaryWriter()

        device = torch.device('cuda')

        # Number of epochs
        num_epochs = 1

        scaler = torch.cuda.amp.GradScaler()
        s, o = self.batch(inputs, batch_size, seq_length)
        for epoch in range(num_epochs):
            for i, (a, b) in enumerate(zip(s, o)):
                a = a.to(device)
                b = b.to(device)

                with torch.cuda.amp.autocast():
                    logits = self.model(a, b)
                    loss = self.lossFn(logits.view(-1, self.model.vocab_size), b.view(-1).long())

                # Zero gradients
                optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}], Loss: {loss.item()}")

                # Log the loss
                writer.add_scalar('Loss/train', loss.item(), epoch * len(inputs) + i)
            
                a = a.cpu()
                b = b.cpu()
                logits = logits.cpu()
                loss = loss.cpu()
                del a, b, logits, loss
                torch.cuda.empty_cache()
                gc.collect()

        writer.close()
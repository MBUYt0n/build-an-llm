import torch
# from torch.utils.tensorboard import SummaryWriter
import gc


class Trainer():
    def __init__(self, func, model):
        self.batch = func
        self.model = model
        self.lossFn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)


    def train(self, inputs, batch_size, seq_length):
        self.model.train()
        # writer = SummaryWriter()

        num_epochs = 1

        s, o = self.batch(inputs, batch_size, seq_length)
        for epoch in range(num_epochs):
            for i, (a, b) in enumerate(zip(s, o)):
                
                logits = self.model(a, b)
                loss = self.lossFn(logits.view(-1, self.model.vocab_size), b.view(-1).long())

                optimizer.zero_grad()

                loss.backward()
                optimizer.step()
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}], Loss: {loss.item()}")

                # writer.add_scalar('Loss/train', loss.item(), epoch * len(inputs) + i)

                logits = logits
                loss = loss
                del a, b, logits, loss
                gc.collect()

        # writer.close()
import torch
import gc
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

class Trainer:
    def __init__(self, model):
        self.model = model
        self.lossFn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, "min", patience=2, factor=0.5
        )

    def train(self, trainLoader, epochs):
        writer = SummaryWriter()
        for epoch in range(epochs):
            self.model.train()
            for i, x in enumerate(tqdm(trainLoader)):
                x = x.to("cuda")
                y = x[:, -1].to("cuda").long()  
                x = x[:, :-1].to("cuda")  
                self.optimizer.zero_grad()
                yHat = self.model(x)  
                yHat = yHat[:, -1, :]
                loss = self.lossFn(yHat, y)
                loss.backward()
                self.optimizer.step()
                writer.add_scalar("loss", loss.item(), epoch * len(trainLoader) + i)
            self.scheduler.step(loss)

            print("epoch", epoch, "loss", loss.item())

            gc.collect()
            torch.cuda.empty_cache()

        writer.close()

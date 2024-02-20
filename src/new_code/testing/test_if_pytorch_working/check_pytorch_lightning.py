import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torch import nn
import pytorch_lightning as pl

# Define a simple neural network module
class SimpleModel(pl.LightningModule):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(5, 2)
        print("INIT ok", flush=True)

    def forward(self, x):
        print("Forward ok", flush=True)
        return F.softmax(self.fc(x), dim=1)

    def training_step(self, batch, batch_idx):
        print("train start ok", flush=True)
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        print("train end ok", flush=True)
        return loss

    def configure_optimizers(self):
        print("optimizers ok", flush=True)
        return torch.optim.Adam(self.parameters(), lr=0.001)

if __name__ == '__main__':
  # Generate some random data
  print("main starting", flush=True)
  x_train = torch.randn(100, 5)
  y_train = torch.randint(0, 2, (100,))

  # Create a DataLoader
  train_dataset = TensorDataset(x_train, y_train)
  train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)

  # Create an instance of the LightningModule
  model = SimpleModel()

  # Create a PyTorch Lightning Trainer
  trainer = pl.Trainer(max_epochs=5, accelerator='gpu', devices=4, strategy='ddp', num_nodes=1)

  # Train the model using the Trainer
  trainer.fit(model, train_loader)
  print("training finished", flush=True)

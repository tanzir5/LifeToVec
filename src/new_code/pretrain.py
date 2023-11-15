import re
import src.transformer
from src.transformer.models import TransformerEncoder
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything, Trainer
import sys
from pathlib import Path
import logging
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import Dataset, DataLoader, random_split
import pickle
import torch
from src.new_code.load_data import CustomDataset

HOME_PATH = str(Path.home())

log = logging.getLogger(__name__)





def last_ckpt(dir_):
    ckpt_path = Path(HOME_PATH, dir_, "last.ckpt")
    if ckpt_path.exists():
        log.info("Checkpoint exists:\n\t%s" %str(ckpt_path))
        return str(ckpt_path)
    else:
        log.info("Checkpoint DOES NOT exists:\n\t%s" %str(ckpt_path))
        return None

def home_path():
    return HOME_PATH

try:
    OmegaConf.register_new_resolver("last_ckpt", last_ckpt)
except Exception as e:
    print(e)


def is_float(string):
    try:
      float(string)
      return True
    except ValueError:
      return False

# Read hparams from the text file
def read_hparams_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        hparams = {}
        for line in lines:
            if len(line) < 2 or line.startswith("#"):
              continue
            #print(line)

            line = line.strip().split('#')[0]

            key, value = line.strip().split(': ')
            value = value.replace('"','')
            if value in ['True', 'False']:
              value = bool(value)
            elif value.isdigit():
              value = int(value)
            elif is_float(value):
              value = float(value)
            hparams[key] = value # float(value) if value.isdigit() else value
            print(key, value)
        return hparams

def get_callbacks():
  callbacks = [
    ModelCheckpoint(
      dirpath='projects/baseball/models/2010',
      filename='model-{epoch:02d}',
      monitor='val_loss_combined',
      save_top_k=2  # Save all models
    )
  ]
  return callbacks

def get_train_val_dataloaders(dataset, batch_size, train_split=0.8, shuffle=True):
  total_samples = len(dataset)
  train_size = int(train_split * total_samples)
  val_size = total_samples - train_size

  train_dataset, val_dataset = random_split(my_dataset, [train_size, val_size])

  return (
    DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle),
    DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
  )

def main():
  hparams_path = 'src/new_code/regular_hparams.txt'
  hparams = read_hparams_from_file(hparams_path)
  model = TransformerEncoder(hparams)
  callbacks = get_callbacks()
  trainer = Trainer(callbacks=callbacks, max_epochs=100)
  with open('projects/baseball/gen_data/mlm.pkl', 'rb') as file:
    dataset = pickle.load(file)

  # Define your batch size
  batch_size = hparams['batch_size']
  
  # Create a data loader
  train_dataloader, val_dataloader = get_train_val_dataloaders(
    dataset=dataset,
    batch_size=batch_size,
  )
  #dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

  print("training and validation dataloaders are created")
  trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == "__main__":
    main()
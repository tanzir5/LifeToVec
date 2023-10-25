import re
import src.transformer
from src.transformer.models import TransformerEncoder
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything, Trainer
import sys
from pathlib import Path
import logging
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import Dataset, DataLoader
import pickle
import torch
from load_data import CustomDataset

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
      dirpath='projects/baseball/models',
      filename='model-{epoch:02d}',
      monitor='loss_total',
      save_top_k=2  # Save all models
    )
  ]
  return callbacks
  

def main():
  hparams_path = 'src/new_code/regular_hparams.txt'
  model = TransformerEncoder(read_hparams_from_file(hparams_path))
  callbacks = get_callbacks()
  trainer = Trainer(callbacks=callbacks, max_epochs=10)
  with open('projects/baseball/gen_data/mlm.pkl', 'rb') as file:
    dataset = pickle.load(file)
  # Define your batch size
  batch_size = 32

  # Create a data loader
  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

  print("ok")
  trainer.fit(model, dataloader)


if __name__ == "__main__":
    main()
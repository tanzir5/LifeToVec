import pandas as pd
import numpy as np
import src.transformer
from src.transformer.models import TransformerEncoder
from src.new_code.load_data import CustomDataset

checkpoint_path = 'projects/baseball/model-epoch=09-v1.ckpt'
model = TransformerEncoder.load_from_checkpoint(checkpoint_path)

model.eval()

with open('projects/baseball/gen_data/mlm.pkl', 'rb') as file:
  dataset = pickle.load(file)

dataset.eval_mode_on()


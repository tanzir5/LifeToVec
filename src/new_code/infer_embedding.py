import pandas as pd
import numpy as np
import src.transformer
from src.transformer.models import TransformerEncoder
from src.new_code.load_data import CustomDataset
from src.new_code.pretrain import read_hparams_from_file
import pickle

# hparams_path = 'src/new_code/regular_hparams.txt'
# hparams = read_hparams_from_file(hparams_path)
# checkpoint_path = 'projects/baseball/models/model-epoch=09-v1.ckpt'
# model = TransformerEncoder.load_from_checkpoint(checkpoint_path, hparams=hparams)

# model.eval()

with open('projects/baseball/gen_data/mlm.pkl', 'rb') as file:
  dataset = pickle.load(file)

print("len", len(dataset))
print(type(dataset))
print(dataset.data["input_ids"].shape)
print(dataset.data["original_sequence"].shape)
dataset.data["input_ids"][:, 0, :] = dataset.data["original_sequence"]
#dataset['input_ids'][0] = dataset['original_sequence']
dataset=dataset[:10]
print(dataset)
print(dataset.data["input_ids"].shape)
print(dataset.data["original_sequence"].shape)
print(dataset.data["sequence_id"].shape)
#dataset.eval_mode_on()


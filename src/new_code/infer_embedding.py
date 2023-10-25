import pandas as pd
import numpy as np
import src.transformer
from src.transformer.models import TransformerEncoder
from src.new_code.load_data import CustomDataset
from src.new_code.pretrain import read_hparams_from_file
from torch.utils.data import DataLoader
import torch

import pickle

hparams_path = 'src/new_code/regular_hparams.txt'
hparams = read_hparams_from_file(hparams_path)
checkpoint_path = 'projects/baseball/models/model-epoch=09-v1.ckpt'
model = TransformerEncoder.load_from_checkpoint(checkpoint_path, hparams=hparams)

model.eval()
devi = str(next(model.parameters()).device)

print(f"Model is on {devi}")
with open('projects/baseball/gen_data/mlm.pkl', 'rb') as file:
  dataset = pickle.load(file)

print("len", len(dataset))
print(type(dataset))
print(dataset.data["input_ids"].shape)
print(dataset.data["original_sequence"].shape)
dataset.data["input_ids"][:, 0, :] = dataset.data["original_sequence"]
batch_size = 32
#dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
all_outputs = []
for i in range(0, len(dataset), batch_size):
    # Move the batch to GPU if available
    batch = dataset[i:i+batch_size]
    if torch.cuda.is_available():
        batch['original_sequence'] = batch['original_sequence'].to('cuda')
        batch['input_ids'] = batch['input_ids'].to('cuda')
        batch['padding_mask'] = batch['padding_mask'].to('cuda')
        batch['target_tokens'] = batch['target_tokens'].to('cuda')
        batch['target_pos'] = batch['target_pos'].to('cuda')
        batch['target_cls'] = batch['target_cls'].to('cuda')
    # Pass the batch through the model
    with torch.no_grad():
        outputs = model(batch)

    # Append the outputs to the list
    all_outputs.append(outputs.cpu())  # Move back to CPU if necessary
    print(type(all_outputs[-1]))
    print(all_outputs[-1])

# Concatenate the outputs along the batch dimension
all_outputs = torch.cat(all_outputs, dim=0)

# 'all_outputs' now contains the model's predictions or outputs for the entire dataset
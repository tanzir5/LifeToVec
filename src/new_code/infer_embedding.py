import pandas as pd
import numpy as np
import src.transformer
from src.transformer.models import TransformerEncoder
from src.new_code.load_data import CustomDataset
from src.new_code.pretrain import read_hparams_from_file
from torch.utils.data import DataLoader
import torch
import json
import pickle
from torch.utils.data._utils.collate import default_collate

def custom_collate(batch):
    collated_batch = {}
    for key in batch[0].keys():
        # Check if the key is 'sequence_id' and handle it separately
        if key == 'sequence_id':
            collated_batch[key] = [item[key] for item in batch]
        else:
            collated_batch[key] = default_collate([item[key] for item in batch])

    return collated_batch

hparams_path = 'src/new_code/regular_hparams.txt'
hparams = read_hparams_from_file(hparams_path)
checkpoint_path = 'projects/baseball/models/2010/model-epoch=28.ckpt'
model = TransformerEncoder.load_from_checkpoint(checkpoint_path, hparams=hparams)
model = model.transformer
model.eval()
devi = str(next(model.parameters()).device)

print(f"Model is on {devi}")
with open('projects/baseball/gen_data/mlm.pkl', 'rb') as file:
  dataset = pickle.load(file)

dataset.set_return_index(True)
print("len", len(dataset))
print(type(dataset))
print(dataset.data["input_ids"].shape)
print(dataset.data["original_sequence"].shape)
dataset.data["input_ids"][:, 0, :] = dataset.data["original_sequence"]
batch_size = 32
#dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=custom_collate)
all_outputs = []
people_embedding_cls = {}
people_embedding_mean = {}
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
      outputs = model(
        x=batch["input_ids"].long(),
        padding_mask=batch["padding_mask"].long(),
      )
    print(f"len(outputs) = {len(outputs)}")
    print(f"batch length = {len(batch['sequence_id'])}")
    print("x"*100)
    for j in range(len(batch['sequence_id'])):
      primary_id = str(batch['sequence_id'][j])
      cls_emb = outputs[j][0].cpu()
      mean_emb = torch.mean(outputs[j],0).cpu()
      people_embedding_cls[primary_id] = cls_emb.tolist()
      people_embedding_mean[primary_id] = mean_emb.tolist()
      print(f"cls_emb dim = {cls_emb.shape}")
      print(f"mean_emb dim = {mean_emb.shape}")
      
    # Append the outputs to the list
    # all_outputs.append(outputs)  # Move back to CPU if necessary
    # print(type(all_outputs[-1]))
    # print(all_outputs[-1])

# Concatenate the outputs along the batch dimension
# all_outputs = torch.cat(all_outputs, dim=0)

# 'all_outputs' now contains the model's predictions or outputs for the entire dataset
file_path = 'projects/baseball/gen_data/cls_embedding_2010.json'
with open(file_path, 'w') as json_file:
    json.dump(people_embedding_cls, json_file)

file_path = 'projects/baseball/gen_data/mean_embedding_2010.json'
with open(file_path, 'w') as json_file:
    json.dump(people_embedding_mean, json_file)
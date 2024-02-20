import pandas as pd
import numpy as np
import src.transformer
from src.transformer.models import TransformerEncoder
from src.new_code.load_data import CustomDataset
from src.new_code.pretrain import read_hparams_from_file
from src.new_code.utils import read_json, print_now
from torch.utils.data import DataLoader
import torch
import json
import pickle
from torch.utils.data._utils.collate import default_collate
import sys
from tqdm import tqdm

def custom_collate(batch):
    collated_batch = {}
    for key in batch[0].keys():
        # Check if the key is 'sequence_id' and handle it separately
        if key == 'sequence_id':
            collated_batch[key] = [item[key] for item in batch]
        else:
            collated_batch[key] = default_collate([item[key] for item in batch])

    return collated_batch

def load_model(checkpoint_path, hparams):
  model = TransformerEncoder.load_from_checkpoint(checkpoint_path, hparams=hparams)
  model = model.transformer
  model.eval()
  device = str(next(model.parameters()).device)
  print_now(f"Model is on {device}")
  return model

def print_now_dataset_stuff(dataset):
  print_now(f"length of dataset {len(dataset)}")
  print_now(f"type of dataset {type(dataset)}")
  print_now(f"input_ids shape = {dataset.data['input_ids'].shape}")
  if "original_sequence" in dataset.data:
    print_now(f"original sequence shape = {dataset.data['original_sequence'].shape}")

def dump_embeddings(path, embeddings_dict):
  with open(path, 'w') as json_file:
    json.dump(embeddings_dict, json_file)


def inference(cfg):
  hparams_path = cfg['HPARAMS_PATH']
  hparams = read_hparams_from_file(hparams_path)
  checkpoint_path = cfg['CHECKPOINT_PATH']
  cls_path = cfg['CLS_WRITE_PATH']
  mean_path = cfg['MEAN_WRITE_PATH']
  tokenized_path = cfg['TOKENIZED_PATH']
  model = load_model(checkpoint_path, hparams)
  with open(tokenized_path, 'rb') as file:
    dataset = pickle.load(file)
  dataset.set_mlm_encoded(False)
  if "original_sequence" in dataset.data:
    dataset.data["input_ids"][:, 0, :] = dataset.data["original_sequence"]
  print_now_dataset_stuff(dataset)

  if 'BATCH_SIZE' in cfg:
    batch_size = cfg['BATCH_SIZE']
  else:
    batch_size = 512
  dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=custom_collate)

  all_outputs = []
  people_embedding_cls = {}
  people_embedding_mean = {}
  for i in tqdm(range(0, len(dataset), batch_size)):
      # Move the batch to GPU if available
      batch = dataset[i:i+batch_size]
      if torch.cuda.is_available():
          batch['input_ids'] = batch['input_ids'].to('cuda')
          batch['padding_mask'] = batch['padding_mask'].to('cuda')
      # Pass the batch through the model
      with torch.no_grad():
        outputs = model(
          x=batch["input_ids"].long(),
          padding_mask=batch["padding_mask"].long(),
        )
      if i%100 == 0:
        print_now(f"printing for batch {i}:")
        print_now(f"len(outputs) = {len(outputs)}")
        print_now(f"batch length = {len(batch['sequence_id'])}")
        
      for j in range(len(batch['sequence_id'])):
        primary_id = str(batch['sequence_id'][j])
        cls_emb = outputs[j][0].cpu()
        mean_emb = torch.mean(outputs[j],0).cpu()
        people_embedding_cls[primary_id] = cls_emb.tolist()
        people_embedding_mean[primary_id] = mean_emb.tolist()
        if i%100 == 0 and j==0:
          print_now(f"cls_emb dim = {cls_emb.shape}")
          print_now(f"mean_emb dim = {mean_emb.shape}")
          assert cls_emb.shape == mean_emb.shape          
      # Append the outputs to the list
      # all_outputs.append(outputs)  # Move back to CPU if necessary
      # print_now(type(all_outputs[-1]))
      # print_now(all_outputs[-1])

  # Concatenate the outputs along the batch dimension
  # all_outputs = torch.cat(all_outputs, dim=0)

  # 'all_outputs' now contains the model's predictions or outputs for the entire dataset
  dump_embeddings(cls_path, people_embedding_cls)
  dump_embeddings(mean_path, people_embedding_mean)


if __name__ == "__main__":
  CFG_PATH = sys.argv[1]
  print_now(CFG_PATH)
  cfg = read_json(CFG_PATH)
  inference(cfg)  



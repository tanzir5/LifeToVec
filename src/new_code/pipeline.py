from src.new_code.custom_vocab import CustomVocabulary
from src.new_code.custom_vocab import DataFile
from src.new_code.create_person_dict import CreatePersonDict
from src.tasks.mlm import MLM
from src.data_new.types import PersonDocument, Background
from src.new_code.load_data import CustomDataset
from src.new_code.utils import get_column_names

import os
import json
import torch
import numpy as np
import pickle 
import fnmatch
import csv
import sys

'''
  1. create vocab
  2. create life_sequence json files
  3. read one by one and run MLM to get mlmencoded documents
'''

PRIMARY_KEY = "primary_key"
DATA_DIRECTORY_PATH = 'data_directory_path'
VOCAB_NAME = 'vocab_name'
VOCAB_WRITE_PATH = 'vocab_write_path'

def read_cfg(path):
  with open(path, 'r') as file:
    cfg = json.load(file)
  return cfg  

def get_raw_file_name(path):
  return path.split("/")[-1].split(".")[0]

def create_vocab(vocab_write_path, data_file_paths, vocab_name, primary_key):
  data_files = []
  for path in data_file_paths:
    data_files.append(DataFile(path, get_raw_file_name(path), primary_key))

  custom_vocab = CustomVocabulary(name=vocab_name, data_files=data_files)
  vocab = custom_vocab.vocab()
  with open(vocab_write_path, 'w') as f:
    json.dump(custom_vocab.token2index, f)



# #create person json files
# creator = CreatePersonDict(file_paths, custom_vocab)
# creator.generate_people_data(write_path)

# #get rid of columns that are unnecessary

# #create mlmencoded documents
# mlm = MLM('dutch_v0', 50)
# mlm.set_vocabulary(custom_vocab)
# input_ids = []
# padding_mask = []
# target_tokens = []
# target_pos = []
# target_cls = []
# with open(write_path, 'r') as f:
#   for line in f:
#     # Parse each line as a JSON-encoded list
#     person_dict = json.loads(line)
  
#     # Now 'json_data' contains the list from the current line
#     # print(type(person_dict))
#     person_document = PersonDocument(
#       person_id=person_dict['person_id'],
#       sentences=person_dict['sentence'],
#       abspos=person_dict['abspos'],
#       age=person_dict['age'],
#       segment=person_dict['segment'],
#       background=Background(**person_dict['background']),
#     )
#     output = mlm.encode_document(person_document)
#     input_ids.append(output.input_ids)
#     padding_mask.append(output.padding_mask)
#     target_tokens.append(output.target_tokens)
#     target_pos.append(output.target_pos)
#     target_cls.append(output.target_cls)

# data = {}
# data['input_ids'] = torch.tensor(np.array(input_ids))
# data['padding_mask'] = torch.tensor(np.array(padding_mask))
# data['target_tokens'] = torch.tensor(np.array(target_tokens))
# data['target_pos'] = torch.tensor(np.array(target_pos))
# data['target_cls'] = torch.tensor(np.array(target_cls))


# print(torch.max(data['input_ids'][:,3]))
# print(data['input_ids'].shape)
# dataset = CustomDataset(data)
# with open('dummy_people_dataset.pkl', 'wb') as file:
#     pickle.dump(dataset, file)


def get_data_files_from_directory(directory, primary_key):
  data_files = []
  for root, dirs, files in os.walk(directory):
    for filename in fnmatch.filter(files, '*.csv'):
      current_file_path = os.path.join(root, filename)
      if primary_key in get_column_names(current_file_path):
        data_files.append(current_file_path)
  return data_files

if __name__ == "__main__":
  CFG_PATH = sys.argv[1]
  print(CFG_PATH)
  cfg = read_cfg(CFG_PATH)

  primary_key = cfg[PRIMARY_KEY]
  # write_path = cfg[WRITE_PATH]
  # sequence_write_path = cfg[SEQUENCE_WRITE_PATH]
  vocab_write_path = cfg[VOCAB_WRITE_PATH]
  vocab_name = cfg[VOCAB_NAME]
  data_file_paths = get_data_files_from_directory(cfg[DATA_DIRECTORY_PATH], primary_key)

  print(len(data_file_paths))
  '''
    1. create vocab
    2. create life_sequence json files
    3. read one by one and run MLM to get mlmencoded documents
  '''
  create_vocab(
    data_file_paths=data_file_paths,
    vocab_write_path=vocab_write_path,
    vocab_name=vocab_name,
    primary_key=primary_key,
  )

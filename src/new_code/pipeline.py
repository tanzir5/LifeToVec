from src.new_code.custom_vocab import CustomVocabulary
from src.new_code.custom_vocab import DataFile
from src.new_code.create_person_dict import CreatePersonDict
from src.tasks.mlm import MLM
from src.data_new.types import PersonDocument, Background
from src.new_code.load_data import CustomDataset
from src.new_code.utils import get_column_names, print_now, read_json
from src.new_code.constants import DAYS_SINCE_FIRST, INF

import os
import json
import torch
import numpy as np
import pickle 
import fnmatch
import csv
import sys
import time

'''
  1. create vocab
  2. create life_sequence json files
  3. read one by one and run MLM to get mlmencoded documents
'''

PRIMARY_KEY = "primary_key"
DATA_DIRECTORY_PATH = 'data_directory_path'
VOCAB_NAME = 'vocab_name'
VOCAB_WRITE_PATH = 'vocab_write_path'
TIME_KEY = "TIME_KEY"
SEQUENCE_WRITE_PATH = "SEQUENCE_WRITE_PATH"
MLM_WRITE_PATH = "MLM_WRITE_PATH"
TIME_RANGE_START = "TIME_RANGE_START"
TIME_RANGE_END = "TIME_RANGE_END"

min_event_threshold = 5



def get_raw_file_name(path):
  return path.split("/")[-1].split(".")[0]

def create_vocab(vocab_write_path, data_file_paths, vocab_name, primary_key):
  data_files = []
  for path in data_file_paths:
    data_files.append(
      DataFile(
        path=path, 
        name=get_raw_file_name(path), 
        primary_key=primary_key,
      )
    )

  custom_vocab = CustomVocabulary(name=vocab_name, data_files=data_files)
  # vocab = custom_vocab.vocab()
  # with open(vocab_write_path, 'w') as f:
  #   json.dump(custom_vocab.token2index, f)
  custom_vocab.save_vocab(vocab_write_path)
  return custom_vocab

def create_person_sequence(file_paths, custom_vocab, write_path, primary_key):
  #create person json files
  creator = CreatePersonDict(
    file_paths=file_paths, 
    primary_key=primary_key, 
    vocab=custom_vocab,
  )
  creator.generate_people_data(write_path)

def get_ids(path):
  with open(path, 'r') as f:
    ids = json.load(f)
  ret_ids = [str(int_id) for int_id in ids]
  return set(ret_ids)

def generate_encoded_data(
  custom_vocab,
  sequence_path,
  write_path,
  time_range=None,
  do_mlm=True,
  needed_ids_path=None,
):
  #create mlmencoded documents
  if needed_ids_path:
    needed_ids = get_ids(needed_ids_path)
    print_now(f'needed ids # = {len(needed_ids)}')
    random_id = list(needed_ids)[0]
    print_now(f'a random id is {random_id}, type is {type(random_id)}')

  mlm = MLM('dutch_v0', 512)
  mlm.set_vocabulary(custom_vocab)
  if time_range:
    mlm.set_time_range(time_range)
  input_ids = []
  padding_mask = []
  target_tokens = []
  target_pos = []
  target_cls = []
  original_sequence = []
  sequence_id = []
  start_time = time.time()
  first_time_print_done = False
  with open(sequence_path, 'r') as f:
    # hardcoding is bad
    # TODO: get # of people beforehand
    total = 27089176
    for i, line in enumerate(f):
      do_print = (i%300000 == 0)
      if i%300000 == 0:
        elapsed_time = time.time() - start_time
        done_fraction = (i+1)/total
        print_now(f"time elapsed: {elapsed_time}, ETA: {elapsed_time/(done_fraction) - elapsed_time}")
        print_now(f"done: {i}")
        print_now(f"done%: {done_fraction*100}")
        print_now(f"included: {len(sequence_id)}")
        print_now(f"included%: {len(sequence_id)/(i+1)*100}")
      
      # Parse each line as a JSON-encoded list
      person_dict = json.loads(line)
      if len(person_dict['sentence']) < min_event_threshold:
        continue
      # Now 'json_data' contains the list from the current line
      # print_now(type(person_dict))
      person_id = person_dict['person_id']
      if first_time_print_done is False:
        print_now(f"first time print")
        print_now(f"person_id is {person_id}, type is {type(person_id)}, present = {person_id in needed_ids}")
        first_time_print_done = True
      if needed_ids_path is not None and person_id not in needed_ids:
        continue
      person_document = PersonDocument(
        person_id=person_dict['person_id'],
        sentences=person_dict['sentence'],
        abspos=[int(x) for x in person_dict['abspos']],
        age=[int(x) for x in person_dict['age']],
        segment=person_dict['segment'],
        background=Background(**person_dict['background']),
      )

      output = mlm.encode_document(
        person_document,
        do_print=do_print,
        do_mlm=do_mlm,
      )

      
      if output is None:
        continue
      sequence_id.append(output.sequence_id)
      original_sequence.append(output.original_sequence)
      input_ids.append(output.input_ids)
      padding_mask.append(output.padding_mask)
      if do_mlm:
        target_tokens.append(output.target_tokens)
        target_pos.append(output.target_pos)
        target_cls.append(output.target_cls)

      if do_mlm and len(sequence_id) >= 1000000:
        break

  print_now(f'total # of people {len(sequence_id)}')  
  data = {}
  data['sequence_id'] = sequence_id
  data['input_ids'] = torch.tensor(np.array(input_ids))
  data['padding_mask'] = torch.tensor(np.array(padding_mask))
  if do_mlm:
    data['target_tokens'] = torch.tensor(np.array(target_tokens))
    data['target_pos'] = torch.tensor(np.array(target_pos))
    data['target_cls'] = torch.tensor(np.array(target_cls))
    data['original_sequence'] = torch.tensor(np.array(original_sequence))
  

  print_now(f"segment max?: {torch.max(data['input_ids'][:,3])}")
  print_now(f"input_ids shape {data['input_ids'].shape}")
  dataset = CustomDataset(data, mlm_encoded=do_mlm)
  with open(write_path, 'wb') as file:
      pickle.dump(dataset, file)




def get_data_files_from_directory(directory, primary_key):
  data_files = []
  for root, dirs, files in os.walk(directory):
    for filename in fnmatch.filter(files, '*.csv'):
      current_file_path = os.path.join(root, filename)
      columns = get_column_names(current_file_path)
      if (
        primary_key in columns and
        ("background" in filename or DAYS_SINCE_FIRST in columns)  
      ): 
        data_files.append(current_file_path)
  return data_files

def get_time_range(cfg):
  time_range = -INF, +INF
  if TIME_RANGE_START in cfg:
    time_range = (cfg[TIME_RANGE_START], time_range[1])
  if TIME_RANGE_END in cfg:
    time_range = (time_range[0], cfg[TIME_RANGE_END])
  return time_range


if __name__ == "__main__":
  CFG_PATH = sys.argv[1]
  print_now(CFG_PATH)
  cfg = read_json(CFG_PATH)

  primary_key = cfg[PRIMARY_KEY]
  # write_path = cfg[WRITE_PATH]
  sequence_write_path = cfg[SEQUENCE_WRITE_PATH]
  vocab_write_path = cfg[VOCAB_WRITE_PATH]
  vocab_name = cfg[VOCAB_NAME]
  time_key = cfg[TIME_KEY]
  mlm_write_path = cfg[MLM_WRITE_PATH]
  data_file_paths = get_data_files_from_directory(
    cfg[DATA_DIRECTORY_PATH], primary_key
  )
  #data_file_paths = data_file_paths[:1]
  print_now(f"# of data_files_paths = {len(data_file_paths)}")
  '''
    1. create vocab
    2. create life_sequence json files
    3. read one by one and run MLM to get mlmencoded documents
  '''
  #custom_vocab = None
  custom_vocab = create_vocab(
    data_file_paths=data_file_paths,
    vocab_write_path=vocab_write_path,
    vocab_name=vocab_name,
    primary_key=primary_key,
  )
  # create_person_sequence(
  #   file_paths=data_file_paths, 
  #   custom_vocab=None,#custom_vocab, 
  #   write_path=sequence_write_path,
  #   primary_key=primary_key,
  # )
  if 'NEEDED_IDS_PATH' in cfg:
    needed_ids_path = cfg['NEEDED_IDS_PATH']
  else:
    needed_ids_path = None
  generate_encoded_data(
    custom_vocab=custom_vocab, 
    sequence_path=sequence_write_path, 
    write_path=mlm_write_path,
    time_range=get_time_range(cfg),
    do_mlm=cfg['DO_MLM'],
    needed_ids_path=needed_ids_path,
  )
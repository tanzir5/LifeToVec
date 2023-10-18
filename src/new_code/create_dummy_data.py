import pandas as pd
import numpy as np
import json
import pickle
from torch.utils.data import Dataset, DataLoader
import torch

from load_data import CustomDataset

vocab_size = 2000
dataset_len = 10000
sequence_len = 500
input_ids = np.random.randint(
    low=1, high=vocab_size, size=(dataset_len, 4, sequence_len
  )
)

padding_lens = np.random.randint(low=0, high=sequence_len-100, size=dataset_len)
padding_mask = np.zeros((dataset_len, sequence_len), dtype=int)
for i in range(dataset_len):
  current_pad_len = padding_lens[i]
  current_original_len = sequence_len - current_pad_len
  padding_mask[i, current_original_len:] = 1

input_ids = input_ids * (1^padding_mask[:, np.newaxis, :])  
input_ids[:, 3] = np.floor(input_ids[:, 3] / vocab_size * 3 + 1)

input_ids = input_ids.tolist()
padding_mask = padding_mask.tolist()

data = {'input_ids':input_ids, 'padding_mask':padding_mask}
file_path = 'dummy_data.json'

with open(file_path, "w") as json_file:
    json.dump(data, json_file)


data['input_ids'] = torch.tensor(data['input_ids'])
data['padding_mask'] = torch.tensor(data['padding_mask'])

print(torch.max(data['input_ids'][:,3]))
dataset = CustomDataset(data)
with open('dummy_dataset.pkl', 'wb') as file:
    pickle.dump(dataset, file)
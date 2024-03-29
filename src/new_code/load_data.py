import json
import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data, mlm_encoded=True):
      self.data = data
      self.set_mlm_encoded(mlm_encoded)

    def set_mlm_encoded(self, mlm_encoded):
      self.mlm_encoded = mlm_encoded
      self.return_index = not self.mlm_encoded
    
    def __len__(self):
        return self.data["input_ids"].shape[0]
    def __reduce__(self):
        return (self.__class__, (self.data,))

    def __getitem__(self, index):
        ret_dict = {            
            "input_ids": self.data["input_ids"][index],
            "padding_mask": self.data["padding_mask"][index],
        }

        if self.mlm_encoded:
          ret_dict.update(
            {
              "original_sequence": self.data["original_sequence"][index],
              "target_tokens": self.data["target_tokens"][index],
              "target_pos": self.data["target_pos"][index],
              "target_cls": self.data["target_cls"][index],
            }
          )

        if self.return_index:
          ret_dict["sequence_id"] = self.data["sequence_id"][index]

        return ret_dict

'''
file_path = 'dummy_data.json'
with open(file_path, "r") as json_file:
    data = json.load(json_file)

data['input_ids'] = torch.tensor(data['input_ids'])
data['padding_mask'] = torch.tensor(data['padding_mask'])


# Create an instance of your custom dataset
dataset = CustomDataset(data)

# Define your batch size
batch_size = 32

# Create a data loader
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Now you can iterate over the dataloader
for batch in dataloader:
    input_ids = batch["input_ids"]  # Shape: (batch_size, 4, 500)
    padding_mask = batch["padding_mask"]  # Shape: (batch_size, 4, 500)
    print(input_ids.shape)
'''
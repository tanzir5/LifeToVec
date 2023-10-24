
import fnmatch
import json
import numpy as np
import os
import pandas as pd 
import sys
import shutil
from src.new_code.utils import get_column_names
from src.new_code.constants import MISSING


SOURCE = 'SOURCE'
DEST = 'DEST'
IMMUTABLE_COLS = 'IMMUTABLE_COLS'
PRIMARY_KEY= 'PRIMARY_KEY'
MAX_UNIQUE_PER_COLUMN = "MAX_UNIQUE_PER_COLUMN"
RESTRICTED_SUBSTRINGS = 'RESTRICTED_SUBSTRINGS'
# Define source and destination directories
def read_cfg(path):
  with open(path, 'r') as file:
    cfg = json.load(file)
  return cfg  

def bucketize(x, current_min, current_max, new_min, new_max):
  try:
    return int(
      np.round(
        (
          (x - current_min) / (current_max - current_min) * (new_max - new_min)
        )
      ) + new_min
    )
  except Exception as e:
    return MISSING

def quantize(col):
  # Find the current min and max values, ignoring NaN values
  current_min = col.min(skipna=True)
  current_max = col.max(skipna=True)
  
  # Define the desired range
  new_min = 1
  new_max = min(current_max, 100)
  assert(current_min != current_max)
  # Perform linear scaling, ignoring NaN values
  scaled_col = col.apply(
    bucketize, 
    args=(current_min, current_max, new_min, new_max)
  )
  return scaled_col

def process_and_write(
  file_path,
  primary_key,
  immutable_cols,
  restricted_substrings,
  max_unique_per_column,
):
  print(file_path)
  print("*"*100)
  df = pd.read_csv(file_path)
  for column in df.columns:
    if column == primary_key or column in immutable_cols: 
      continue
    elif (
      any(item in column for item in restricted_substrings) or 
      (
        df[column].dtype=='object' and 
        len(df[column].unique()) > max_unique_per_column
      )
    ):
      print("dropping column", column)
      df.drop(columns=[column], inplace=True)
    elif np.issubdtype(df[column].dtype, np.number):
      print("quantizing column", column)
      df[column] = quantize(df[column])
  df.fillna(value=MISSING, inplace=True)
  df.to_csv(file_path, index=False)

if __name__ == "__main__":
  CFG_PATH = sys.argv[1]
  print(CFG_PATH)
  cfg = read_cfg(CFG_PATH)

  source_dir = cfg[SOURCE]
  destination_dir = cfg[DEST]
  immutable_cols = cfg[IMMUTABLE_COLS]
  primary_key = cfg[PRIMARY_KEY]
  restricted_substrings = cfg[RESTRICTED_SUBSTRINGS]
  max_unique_per_column = cfg[MAX_UNIQUE_PER_COLUMN]
  # Use shutil.copytree() to copy the entire directory recursively
  if os.path.exists(destination_dir):
    shutil.rmtree(destination_dir)
  shutil.copytree(source_dir, destination_dir)
  for root, dirs, files in os.walk(destination_dir):
    for filename in fnmatch.filter(files, '*.csv'):
      current_file_path = os.path.join(root, filename)
      if primary_key in get_column_names(current_file_path):
        process_and_write(
          file_path=current_file_path,
          primary_key=primary_key,
          immutable_cols=immutable_cols,
          restricted_substrings=restricted_substrings,
          max_unique_per_column=max_unique_per_column,
        )
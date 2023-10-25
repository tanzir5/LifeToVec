
import fnmatch
import json
import numpy as np
import os
import pandas as pd 
import sys
import shutil
from src.new_code.utils import get_column_names
from src.new_code.constants import MISSING, DAYS_SINCE_FIRST, FIRST_EVENT_TIME, AGE

TIME_KEY = 'TIME_KEY'
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

all_cols = {}
all_primaries = set()
def process_and_write(
  file_path,
  primary_key,
  immutable_cols,
  restricted_substrings,
  max_unique_per_column,
  time_key,
  first_event_time,
):
  global all_primaries
  print(file_path)
  print("*"*100)
  df = pd.read_csv(file_path)
  all_primaries = all_primaries | set(df[primary_key].unique())
  col_for_drop_check = DAYS_SINCE_FIRST
  if DAYS_SINCE_FIRST not in df.columns:
    col_for_drop_check = time_key
  
  df.dropna(subset=[col_for_drop_check], inplace=True)
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
  for col in df.columns:
    if col not in all_cols:
      all_cols[col] = []
    all_cols[col].append(file_path)
  if DAYS_SINCE_FIRST not in df.columns:
    df[DAYS_SINCE_FIRST] = df[time_key].apply(lambda x: x-first_event_time)
    df.drop(columns=[time_key], inplace=True)

  df[AGE] = 0
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
  if TIME_KEY in cfg:
    time_key = cfg[TIME_KEY]
    first_event_time = cfg[FIRST_EVENT_TIME]
  else:
    time_key = None
  # Use shutil.copytree() to copy the entire directory recursively
  if os.path.exists(destination_dir):
    shutil.rmtree(destination_dir)
  shutil.copytree(source_dir, destination_dir)
  for root, dirs, files in os.walk(destination_dir):
    for filename in fnmatch.filter(files, '*.csv'):
      current_file_path = os.path.join(root, filename)
      cols = get_column_names(current_file_path)
      if (
        primary_key in cols and (DAYS_SINCE_FIRST in cols or time_key in cols)
      ):
        print(filename)
        process_and_write(
          file_path=current_file_path,
          primary_key=primary_key,
          immutable_cols=immutable_cols,
          restricted_substrings=restricted_substrings,
          max_unique_per_column=max_unique_per_column,
          time_key=time_key,
          first_event_time=first_event_time
        )

  for col in all_cols:
    print(col, ":", len(all_cols[col]), "\n", all_cols[col])

  print(f"# of people {len(all_primaries)}")
  df = pd.read_csv('projects/baseball/data/baseballdatabank-2023.1 2/core/People.csv')
  people_have = set(df[primary_key].unique())
  print(f"intersection size = {len(people_have & all_primaries)}")
  print(f"subtract size 1 = {len(people_have - all_primaries)}")
  print(f"subtract size 2 = {len(all_primaries & people_have)}")
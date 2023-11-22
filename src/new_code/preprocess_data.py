
import fnmatch
import json
import math
import numpy as np
import os
import pandas as pd 
import sys
import shutil
from src.new_code.utils import get_column_names, print_now
from src.new_code.constants import MISSING, DAYS_SINCE_FIRST, FIRST_EVENT_TIME, AGE, INF
from functools import partial


TIME_KEY = 'TIME_KEY'
SOURCE = 'SOURCE'
DEST = 'DEST'
IMMUTABLE_COLS = 'IMMUTABLE_COLS'
PRIMARY_KEY= 'PRIMARY_KEY'
MAX_UNIQUE_PER_COLUMN = "MAX_UNIQUE_PER_COLUMN"
RESTRICTED_SUBSTRINGS = 'RESTRICTED_SUBSTRINGS'
CALCULATE_AGE = 'CALCULATE_AGE'
DELIMITER = 'DELIMITER'
MISSING_SIGNIFIERS = 'MISSING_SIGNIFIERS'
# Define source and destination directories
def read_cfg(path):
  with open(path, 'r') as file:
    cfg = json.load(file)
  return cfg  

def bucketize(x, current_min, current_max, new_min, new_max, missing_signfiers):
  try:
    if x in missing_signfiers:
      return MISSING
    return int(
      np.round(
        (
          (x - current_min) / (current_max - current_min) * (new_max - new_min)
        )
      ) + new_min
    )
  except Exception as e:
    if math.isnan(float(x)) is False:
      print_now(e)
      print_now(x, current_min, current_max, new_min, new_max)
    return MISSING

def quantize(col, name):
  # Find the current min and max values, ignoring NaN values
  current_min = col.min(skipna=True)
  current_max = np.ceil(col.max(skipna=True))

  if current_min == -np.inf:
    current_min = int(-INF)
  if current_max == np.inf:
    current_max = int(INF)
  
  # Define the desired range
  new_min = 1
  new_max = min(current_max, 100)
  assert(current_min != current_max)
  # Perform linear scaling, ignoring NaN values
  scaled_col = col.apply(
    bucketize, 
    args=(current_min, current_max, new_min, new_max, missing_signfiers)
  )
  ok_num = len(scaled_col) - np.sum(scaled_col=='MISSING')
  print_now(f"column = {name}, ok count = {ok_num}, ok % = {ok_num/len(scaled_col) * 100}")
  assert(ok_num > (len(scaled_col)/100))
  return scaled_col, ok_num > (len(scaled_col)/100)

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
  calculate_age,
  missing_signfiers
):
  global all_primaries
  if 'background' in file_path:
    print_now(f"background file {file_path} is left as it is.")
    return
  print_now(file_path)
  print_now("*"*100)
  df = pd.read_csv(file_path, delimiter=delimiter)
  all_primaries = all_primaries | set(df[primary_key].unique())
  col_for_drop_check = DAYS_SINCE_FIRST
  if DAYS_SINCE_FIRST not in df.columns:
    col_for_drop_check = time_key
  if calculate_age is False:
    if AGE not in df.columns:
      print_now(f"deleting {file_path} as it does not have the column age.")
      os.remove(file_path)
      return
    else:
      df[AGE] = df[AGE].to_numpy(int) 
  else:
    # TODO: write logic for calculating age at event
    df[AGE] = 0

  df.dropna(subset=[col_for_drop_check], inplace=True)
  for column in df.columns:
    if column in [AGE, DAYS_SINCE_FIRST, time_key]:
      df[column] = np.round(df[column].tolist()).astype(int)
    elif column == primary_key or column in immutable_cols: 
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
    elif np.issubdtype(df[column].dtype, np.number) and len(df[column].unique()) > 100:
      print("quantizing column", column)
      scaled_col, have_enough = quantize(df[column], column)
      if have_enough:
        df[column] = scaled_col
    else:
      print(f"keeping categorical column {column} with {len(df[column].unique())} items")
  df.fillna(value=MISSING, inplace=True)
  for col in df.columns:
    if col not in all_cols:
      all_cols[col] = []
    all_cols[col].append(file_path)
  if DAYS_SINCE_FIRST not in df.columns:
    df[DAYS_SINCE_FIRST] = df[time_key].apply(lambda x: x-first_event_time)
    df.drop(columns=[time_key], inplace=True)
  else:
    df[DAYS_SINCE_FIRST] = df[DAYS_SINCE_FIRST].to_numpy(int)
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
  calculate_age = cfg[CALCULATE_AGE]
  if DELIMITER in cfg:
    delimiter = cfg[DELIMITER]
  else:
    delimiter = ','
  if TIME_KEY in cfg:
    time_key = cfg[TIME_KEY]
    first_event_time = cfg[FIRST_EVENT_TIME]
  else:
    time_key = None
  if MISSING_SIGNIFIERS in cfg:
    missing_signfiers = cfg[missing_signfiers]
  else:
    missing_signfiers = []
  # Use shutil.copytree() to copy the entire directory recursively
  if os.path.exists(destination_dir):
    shutil.rmtree(destination_dir)
  shutil.copytree(source_dir, destination_dir)
  for root, dirs, files in os.walk(destination_dir):
    for filename in fnmatch.filter(files, '*.csv'):
      current_file_path = os.path.join(root, filename)
      cols = get_column_names(current_file_path, delimiter=delimiter)
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
          first_event_time=first_event_time,
          calculate_age=calculate_age,
          missing_signfiers=missing_signfiers
        )

  for col in all_cols:
    print(col, ":", len(all_cols[col]), "\n", all_cols[col])

  print(f"# of people {len(all_primaries)}")

  # df = pd.read_csv('projects/baseball/data/baseballdatabank-2023.1 2/core/People.csv')
  # people_have = set(df[primary_key].unique())
  # print(f"intersection size = {len(people_have & all_primaries)}")
  # print(f"subtract size 1 = {len(people_have - all_primaries)}")
  # print(f"subtract size 2 = {len(all_primaries & people_have)}")
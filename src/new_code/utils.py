import csv
import pandas as pd
import json
from functools import partial

print_now = partial(print, flush=True)

def get_column_names(csv_file, delimiter=','):
  df = pd.read_csv(csv_file, delimiter=delimiter, nrows=2)
  return df.columns

def read_json(path):
  with open(path, 'r') as file:
    data = json.load(file)
  return data  
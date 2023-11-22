import csv
import pandas as pd
from functools import partial

print_now = partial(print, flush=True)

def get_column_names(csv_file, delimiter=','):
  df = pd.read_csv(csv_file, delimiter=delimiter, nrows=2)
  return df.columns


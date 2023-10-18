import pandas as pd
import numpy as np
import os

bad_names = ['readme', 'README', 'DS_Store']

player_years = {}
directory = 'data/baseball_data/'
player_csv_count = 0
total_count = 0
mn = 1e9 
mx = -1e9
lens = 0
cols = set()
for root, dirs, files in os.walk(directory):
  for file in files:
    file_path = os.path.join(root, file)
    found_bad = False
    for x in bad_names:
      if x in file_path:
        found_bad = True
    if found_bad:
      continue
    df = pd.read_csv(file_path)
    if 'player_ID' in df.columns:
      df['playerID'] = df['player_ID']
    if 'year_ID' in df.columns:
      df['yearID'] = df['year_ID']
    
    if 'playerID' in df.columns and 'yearID' in df.columns:
      print(file_path, len(df))
      mn = df['yearID'].min()
      mx = df['yearID'].max()
      cols  |= set(df.columns)
      current_unique_players = df['playerID'].unique()
      for player_id in current_unique_players:
        current_player_years = df[df['playerID']==player_id]['yearID']
        min_year = current_player_years.min()
        max_year = current_player_years.max()
        if player_id not in player_years:
          player_years[player_id] = (min_year, max_year)
        else:
          old_min_year, old_max_year = player_years[player_id]
          min_year = min(old_min_year, min_year)
          max_year = max(old_max_year, max_year)
          player_years[player_id] = (min_year, max_year)

print(mn, mx, lens)
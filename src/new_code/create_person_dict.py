from src.new_code.constants import GENDER, BIRTH_MONTH, BIRTH_YEAR, ORIGIN, DAYS_SINCE_FIRST, AGE, DELIMITER, IGNORE_COLUMNS, MISSING
from src.data_new.types import Background, PersonDocument
from src.new_code.utils import print_now
import pandas as pd
import json

class CreatePersonDict():
  def __init__(self, file_paths, primary_key, vocab=None, vocab_path=None):
    self.source_paths = file_paths
    self.background_file_path = self.get_background_file(file_paths)
    self.source_paths.remove(self.background_file_path)
    self.vocab = self.get_vocab(vocab, vocab_path)
    self.primary_key = primary_key

  def get_vocab(self, vocab, vocab_path):
    if vocab is not None:
      return vocab
    elif vocab_path is not None:
      return load(vocab_path) # ???
    else:
      return None
      #raise ValueError("Both vocab and vocab_path cannot be None")


  def get_background_file(self, file_paths):
    background_file_path = [fp for fp in file_paths if 'background' in fp]
    assert(len(background_file_path) == 1)
    return background_file_path[0]

  def initialize_backgrounds(self):
    people = {}
    background_df = pd.read_csv(
      self.background_file_path,
      dtype=str,
      delimiter=DELIMITER
    )
    background_df = background_df.fillna(MISSING)
    print_now(f"{len(background_df)} people in background file")
    # for index, row in background_df.iterrows():
    #   if index%100000 == 0:
    #     print_now(f"done = {index} people")
    #     print_now(f"done% = {index/len(background_df)*100}")
    #   person_id = row[self.primary_key]
    #   person = {
    #     'person_id': person_id, 
    #     'background': {
    #       'origin': f"{ORIGIN}_{row[ORIGIN]}",
    #       'gender': f"{GENDER}_{row[GENDER]}",
    #       'birth_month': f"{BIRTH_MONTH}_{row[BIRTH_MONTH]}",
    #       'birth_year': f"{BIRTH_YEAR}_{row[BIRTH_YEAR]}",     
    #     },
    #     'events': []
    #   }
    #   people[person_id] = person

    print_now(f"columns in order = {background_df.columns}")
    for row in background_df.itertuples(name=None):
      index = row[0]
      person_id = row[1]
      person = {
        'person_id': person_id, 
        'background': {
          'birth_year': f"{BIRTH_YEAR}_{row[2]}",  
          'birth_month': f"{BIRTH_MONTH}_{row[3]}",  
          'gender': f"{GENDER}_{row[4]}", 
          'origin': f"{ORIGIN}_{row[5]}",
        },
        'events': []
      }  
      people[person_id] = person
    
      if index%100000 == 0:
        print_now(f"done = {index} people")
        print_now(f"done% = {index/len(background_df)*100}")
        print_now(f"latest person\n{person}")
    
    return people

  # def format_event_for_tokenization(self, event):
  #   sentence = []
  #   for attribute in event.index:
  #     if attribute not in [self.primary_key, DAYS_SINCE_FIRST, AGE]:
  #       sentence.append(f"{attribute}_{str(event[attribute])}")
  #   return sentence

  def format_event_for_tokenization(self, event):
    sentence = []
    for key, value in event.items():
      if key not in [self.primary_key, DAYS_SINCE_FIRST, AGE, 'Index']:
        sentence.append(f"{key}_{str(value)}")
    return sentence

  def _make_int(self, value):
    return round(float(value))

  def expand_person(self, person):
    sentences = []
    abspos = []
    ages = []
    segments = []
    for event in person['events']:
      sentences.append(self.format_event_for_tokenization(event))
      abspos.append(event[DAYS_SINCE_FIRST])
      ages.append(event[AGE])
      if len(abspos) > 1 and abspos[-2] == abspos[-1]:
        segments.append(1)
      else:
        segments.append(2)
    return sentences, abspos, ages, segments

  def write_people_data(self, people, write_path):
    with open(write_path, 'w') as f:
      for _, person_data in people.items():
        json.dump(person_data, f)
        f.write('\n')

  def generate_people_data(self, write_path):
    self.people = self.initialize_backgrounds()
    dataframes = []
    for file_count, source_path in enumerate(self.source_paths):
      print_now(f"working with {source_path}")

      df = pd.read_csv(
        source_path,
        dtype=str,
        delimiter=DELIMITER,
        usecols=lambda column: column not in IGNORE_COLUMNS,
      )
      print_now(f"# of records in this file = {len(df)}")
      print_now(f"done = {file_count}, remaining = {len(self.source_paths)-file_count}")
      # df[AGE] = df[AGE].apply(lambda x: self._make_int(x))
      # df[DAYS_SINCE_FIRST] = df[DAYS_SINCE_FIRST].apply(lambda x: self._make_int(x))
      
      # for _, row in df.iterrows():
      #   person_id = row[self.primary_key]
      #   if person_id in self.people:
      #     self.people[person_id]['events'].append(row)


      for row in df.itertuples():
        row = row._asdict()
        person_id = row[self.primary_key]
        if person_id in self.people:
          self.people[person_id]['events'].append(row)
    
    for index, (key, value) in enumerate(self.people.items()):
      self.people[key]['events'] = sorted(
        value['events'], 
        key=lambda x: x[DAYS_SINCE_FIRST]
      )
      sentence, abspos, age, segment = self.expand_person(self.people[key])
      self.people[key] = {
        'person_id': value['person_id'],
        'background': value['background'],
        'sentence': sentence,
        'abspos': abspos,
        'age': age,
        'segment': segment,
      }
      if index%100000 == 0:
        print_now(f"done = {index} people")
        print_now(f"done% = {index/len(self.people)*100}")
        
    self.write_people_data(self.people, write_path)

import json

from src.data_new.types import Background, PersonDocument


def get_person_from_dict(x):
  return PersonDocumet(**x)


def get_all_persons_from_json(json_path):
  with open(json_path, 'r') as json_file:
    for line in json_file:
        person_dict = json.loads(line) 
        # yield get_person_from_dict(person_dict)
        yield PersonDocument(**x)

# for item in read_large_list_from_json("large_data.json"):
#   print(item) 


def write_all_persons_to_json(json_path):
  with open(json_path, 'w') as json_file:
    for i in range(1000):
      a = [j for j in range(i, i+10)]
      json.dump(a, json_file)
      json_file.write('\n')




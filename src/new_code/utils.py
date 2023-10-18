import csv
def get_column_names(csv_file):
  with open(csv_file, 'r', newline='') as file:
    reader = csv.reader(file)
    header = next(reader)
    return header
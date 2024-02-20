'''
Fixes
1. make get_distance a function and pass person_embedding, at least embedding lookup and tensor casting for person will be done only once instead of 100 times.
    brings down to 1.3 seconds from 26 seconds.
2. get rid of tensor typecasting (less important)
3. if distance > partner_distance: (less important)
     partner_rank += 1
4. call rng.choice() only once at the beginning. 
    brings down to 1.8 seconds from 13.2 seconds
'''

'''
Failed fixes
1. Pythonic > is slightly slower than sort at least here. But keeping it since it is easy to read.
2. rng.choice() and np.random.choice() are the same
3. sample.remove and sample.pop are not time-consuming
4. 
'''

'''
1
n = 10000
marriage rank time: 1.2935118675231934 seconds
marriage rank time: 26.08594298362732 seconds
'''

'''
2
n = 10000

marriage rank time: 26.08594298362732 seconds
marriage rank time: 24.87499690055847 seconds
'''


'''
3
n = 40000

marriage rank time: 13.29706597328186 seconds
marriage rank time: 12.934809923171997 seconds
'''

'''
4
n = 40000

marriage rank time: 13.29706597328186 seconds
marriage rank time: 1.7928481101989746 seconds
'''




import numpy as np
# from torch import Tensor
from sentence_transformers import util
from time import time

def get_distance(person, partner, distance_matrix, embedding_dict, person_embedding):
  distance = None
  #person_embedding = embedding_dict[person]
  if person in distance_matrix:
    if partner in distance_matrix[person]:
      distance = distance_matrix[person][partner]
  if partner in distance_matrix:
    if person in distance_matrix[partner]:
      distance = distance_matrix[partner][person]

  if distance is None:
    partner_embedding = embedding_dict[partner]
    distance = util.cos_sim(person_embedding, partner_embedding).numpy()[0][0]
    if person not in distance_matrix:
      distance_matrix[person] = {}
    distance_matrix[person][partner] = distance
  
  return distance

def get_marriage_rank_by_year(embedding_dict, distance_matrix, full_male_list, full_female_list, full_user_set, marriage_data, gender_map):

  full_user_set = set(embedding_dict.keys())

  ################

  reduced_male_set = set(full_male_list).intersection(full_user_set)
  reduced_male_list = list(reduced_male_set)


  reduced_female_set = set(full_female_list).intersection(full_user_set)
  reduced_female_list = list(reduced_female_set)


  yearly_rank_averages = {}
  overall_ranks = []

  rng = np.random.default_rng()

  years = len(marriage_data)
  max_marriages_per_year = np.max([len(marriage_data[year]) for year in marriage_data])

  male_sample = rng.choice(reduced_male_list, size=(years, max_marriages_per_year, 101), replace=True)
  female_sample = rng.choice(reduced_female_list, size=(years, max_marriages_per_year, 101), replace=True)
  for year_idx, year in enumerate(marriage_data):
    partner_ranks = []
    yearly_data = marriage_data[year]
    for person_idx, person in enumerate(yearly_data):
      if person not in full_user_set:
        continue
      partner = yearly_data[person]
      if partner not in full_user_set:
        continue


      person_embedding = embedding_dict[person]
      distance = get_distance(person, partner, distance_matrix, embedding_dict, person_embedding)
    

      partner_distance = distance
      partner_rank = 1

      gender = gender_map[partner]

      if gender == 1:
        sample = list(male_sample[year_idx][person_idx])
      if gender == 2:
        sample = list(female_sample[year_idx][person_idx])

      if person in sample:
        sample.remove(person)
      else:
        sample.pop()

      for other in sample:
        distance = get_distance(person, other, distance_matrix, embedding_dict, person_embedding)
        if distance > partner_distance:
          partner_rank += 1


      partner_ranks.append(partner_rank)
      overall_ranks.append(partner_rank)

    yearly_rank_averages[int(year)] = np.mean(partner_ranks)

  yearly_rank_averages['OVERALL'] = np.mean(overall_ranks)
  return yearly_rank_averages


def create_data(n = int(15e4)):
  embedding_dict = {}
  embs = np.random.rand(n, 64)
  for i, emb in enumerate(embs):
    embedding_dict[i] = emb
  distance_matrix = {}
  random_bool_array = np.array(np.random.choice([True, False], size=n))
  full_male_list = random_bool_array.nonzero()[0]
  full_female_list = (~random_bool_array).nonzero()[0]
  full_user_set = np.arange(0,n)

  gender_map = {}
  for x, y in enumerate(random_bool_array):
    gender_map[x] = 1 if y==True else 2

  marriage_data = {}
  for year in range(2011, 2022):
    folks = np.random.choice(full_user_set, size=int(n/25))
    others = np.random.choice(full_user_set, size=int(n/25))
    marriage_data[year] = {}
    for x, y in zip(folks, others):
      marriage_data[year][x] = y

  return embedding_dict, distance_matrix, full_male_list, full_female_list, full_user_set, marriage_data, gender_map

if __name__ == '__main__':
  n = 1000
  start_time = time()
  embedding_dict, distance_matrix, full_male_list, full_female_list, full_user_set, marriage_data, gender_map = create_data(n)
  elapsed_time = time() - start_time
  print("data generation time:", elapsed_time, "seconds")
  start_time = time()      
  get_marriage_rank_by_year(embedding_dict, distance_matrix, full_male_list, full_female_list, full_user_set, marriage_data, gender_map)
  elapsed_time = time() - start_time
  print("marriage rank time:", elapsed_time, "seconds")
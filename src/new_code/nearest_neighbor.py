from sentence_transformers import util
from torch import Tensor
from typing import Dict, Union, List, Any

import torch

# pid means primary_id or primary_key in the context of this code. 
# pid should be string

def build_index(
  embs_dict: Dict[str, Tensor]
) -> Dict[str, Union[Tensor, Dict]]:
  """Creates a helper index that maps string keys to integer indices"""

  embs = []
  idx_to_pid = {}
  pid_to_idx = {}
  for i, (key, value) in enumerate(embs_dict.items()):
    embs.append(value)
    idx_to_pid[i] = key
    pid_to_idx[key] = i
  
  return {
    'embedding':torch.tensor(embs), 
    'idx_to_pid':idx_to_pid,
    'pid_to_idx':pid_to_idx,
  }


def get_nearest_neighbor(
  corpus_data: Dict[str, Union[Tensor, Dict]], 
  query_data: Dict[str, Union[Tensor, Dict]], 
  top_k: int = 10,
  check_pids: Union[List[str], None] = None,
  ignore_self: bool = True,
) -> Dict[str, List]:
  """Finds the top k nearest neighbors of each query in corpus.

  Args:
    corpus_data: this should be a dictionary returned from calling build_index()
      containing all the database/corpus embeddings.
    query_data: similar to corpus_data but for queries.
    top_k: top k nearest neighbors will be returned.
    check_pids: these special pids, their ranks, and similarity will be 
      returned for each query even when they are not one of the top k nearest 
      neighbors.
    ignore_self: decides if the query result ignores pids same as the query pid 

  Returns:
    A dictionary mapping pids to the list of nearest neighbors along with the
    score and rank. The list is sorted in increasing order of rank.
  """
  if check_pids is None:
    check_pids = []
    find_all = False
  else:
    find_all = True
  
  results = util.semantic_search(
    query_embeddings=query_data['embedding'],
    corpus_embeddings=corpus_data['embedding'],
    top_k=top_k if not find_all else len(corpus_data['embedding']),
  )
  result_pids = {}
  for i, result in enumerate(results):
    pid = query_data['idx_to_pid'][i]
    result_pids[pid] = []
    for rank, neighbor in enumerate(result):
      neighbor_pid = corpus_data['idx_to_pid'][neighbor['corpus_id']]
      if ignore_self and neighbor_pid == pid:
        top_k += 1
        continue
      if rank < top_k or neighbor_pid in check_pids:
        result_pids[pid].append({
            'pid': neighbor_pid,
            'score': neighbor['score'],
            'rank': rank,
          }
        )

  return result_pids


def get_nearest_neighbor_e2e(
  corpus_embs_dict: Dict[str, Tensor],
  query_embs_dict: Dict[str, Tensor],
  top_k: int = 10,
  check_pids: Union[List[str], None] = None,
  ignore_self: bool = True,
  return_index: bool = False,
) -> Dict[str,Any]:
  """Finds the top k nearest neighbors of each query in corpus.

  Args:
    corpus_embs_dict: this is a dictionary mapping corpus pids to embeddings.
    query_embs_dict: similar to corpus_embs_dict but for queries.
    top_k: top k nearest neighbors will be returned.
    check_pids: these special pids, their ranks, and similarity will be 
      returned for each query even when they are not one of the top k nearest 
      neighbors.
    ignore_self: decides if the query result ignores pids same as the query pid 
    return_index: decides if the indexes built for corpus and query
      should be returned or not.

  Returns:
    A dictionary with potentially three keys,
      'result': a dictionary mapping pids to the list of nearest neighbors
        along with the score and rank. The list is sorted in increasing order of
        rank.
      'corpus_index': the index built for corpus embeddings
      'query_index': the index built for query embeddings
  """  
  corpus_index = build_index(corpus_embs_dict)
  query_index = build_index(query_embs_dict)
  ret_dict = {
    'result': get_nearest_neighbor(
      corpus_data=corpus_index, 
      query_data=query_index, 
      top_k=top_k,
      check_pids=check_pids,
      ignore_self=ignore_self,
    )
  }
  if return_index:
    ret_dict['corpus_index'] = corpus_index
    ret_dict['query_index'] = query_index
  return ret_dict
  

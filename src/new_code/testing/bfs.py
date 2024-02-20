import pickle
import pandas as pd
import numpy as np
import random
import time

def bfs(person, num_steps):

    seen = set()
    
    ret_list = []
    
    connections = adjacency_dict[person]
    seen.add(person)
    for i in range(num_steps):
    
        new_additions = []
        if len(connections) > 2:
            connections = random.sample(connections, 2)
        for connection in connections:
        
            seen.add(connection)
                
            ret_list.append((connection, i+1))
            
            further = adjacency_dict[connection]
            for other_connection in further:
                if other_connection not in seen:
                    new_additions.append(other_connection)
                    
        connections = new_additions
        
    return ret_list
    
########################################################################################

adjacency_dict = {
  'A': ['B', 'C'],
  'B': ['A', 'C'],
  'C': ['B', 'A'],
}

result = bfs('A', 3)
print(result)
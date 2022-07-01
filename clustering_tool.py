import math
import numpy as np
import pandas as pd

def identify_key_elements(d_current, c_current):
    # Finding first key element which has the least average distance from others
    first_key = math.inf
    first_key_index = int()
    m = len(d_current)
    s = set()
    k = list(np.arange(0,len(d_current),1))
    for i in range(0,len(d_current)):
        avg = 0
        avg = sum(d_current.iloc[i].tolist())/m
        if(avg<first_key):
            first_key = avg
            first_key_index = i
    k.remove(first_key_index)
    s.add(first_key_index)
    n = 1
    # Doing iterations until we have enough clusters
    while(n!=c_current):
        next_key_index = int()
        min_dist = math.inf
        max_dist = -math.inf
        # Finding next key element which has the most minimums distance to the current key elements
        for index in k:
            for j in range(0,len(s)):
                if(d_current.iloc[index,j]<min_dist):
                    min_dist = d_current.iloc[index,j]
            if(min_dist>max_dist):
                max_dist = min_dist
                next_key_index = index
        k.remove(next_key_index)
        s.add(next_key_index)
        n = n + 1
    # Returning all key elements(aka new clusters)
    return s
       
import pandas as pd
import math
import numpy as np
from scipy.spatial import distance_matrix


def identify_key_elements(D_current: pd.DataFrame, C_current: int)->pd.DataFrame:
    
    # Finding first key element which has the least average distance from others
    first_key = math.inf
    first_key_index = int()
    D_current_list = D_current.values.tolist()
    m = len(D_current_list)
    s = set()
    k = list(np.arange(0,m,1))
    for i in range(m):
        avg = 0
        avg = sum(D_current_list[i])/m
        if(avg<first_key):
            first_key = avg
            first_key_index = i
    k.remove(first_key_index)
    s.add(first_key_index)
    n = 1
    
    # Doing iterations until we have enough clusters
    while(n < C_current):
    
        next_key_index = int()
        min_dist = math.inf
        max_dist = -math.inf
    
        # Finding next key element which has the most minimums distance to the current key elements
        for index in k:
            min_dist = math.inf
            for j in range(0,len(s)):
                dist = D_current_list[index][j]
                if(dist < min_dist):
                    min_dist =dist
            if(min_dist > max_dist):
                max_dist = min_dist
                next_key_index = index
        k.remove(next_key_index)
        s.add(next_key_index)
        n = n + 1
    
    # Returning all key elements(aka new clusters)
    return pd.DataFrame(s, columns=['Clusters'])


def custom_clustering(X_train: pd.DataFrame, C_target: int)-> pd.DataFrame:

    # Decleration of the algorithm's parameters.
    # DESIRED_CLUSTERS = 3
    CONSTANT_NUMBER_G = 10
    CONSTATN_NUMBER_K = 5

    ## 1
    # Computing distance matrix(D_original).
    print('Computing distance matrix(D_original)\n')
    D_original = pd.DataFrame(distance_matrix(X_train.values,X_train.values), index=list(range(X_train.shape[0])), columns=list(range(X_train.shape[0])))
    print('Computing distance matrix(D_original) completed\n')
    ##2
    # Finding K-Nearest data points.
    print('Finding K-Nearest data points\n')
    K = CONSTATN_NUMBER_K
    R_k = []
    for idx ,x in D_original.iterrows():
        R_k.append(list(x.nsmallest(K + 1).keys()))
    R_k = pd.DataFrame(R_k)
    print('Finding K-Nearest data points completed\n')

    ##3
    # Initializing cluster labels.
    print('Initializing cluster labels\n')
    N = X_train.shape[0]
    L = list(range(0,N))
    G = CONSTANT_NUMBER_G
    print('Initializing cluster labels completed\n')
    
    ##4
    # Initializing algorithm params.
    C_previous = N
    C_current = math.floor(N/G)
    # C_target = DESIRED_CLUSTERS

    ##5
    # Initial Computation of D_current 
    print('Initial Computation of D_current\n')
    D_current = np.zeros((N,N))
    D_org_list = D_original.values.tolist()
    R_k_list = R_k.values.tolist()
    k2 = (K + 1)**2
    for i in range(N):
        for j in range(i, N):
            sum_val = 0
            R_k_i = R_k_list[i]
            R_k_j = R_k_list[j]
            for h in range(K):
                for p in range(K):                
                    sum_val += D_org_list[R_k_i[h]][R_k_j[p]]
            dist = sum_val/(k2)
            D_current[i][j] = dist
            D_current[j][i] = dist
    D_current = pd.DataFrame(D_current)
    print("D_current: ", D_current.shape)
    print(D_current.head())

    print('Initial Computation of D_current completed\n')
    ##6
    # Main loop of the algorithm

    print('Main loop of the algorithm\n')
    while(C_current > C_target):
        S_current = identify_key_elements(D_current, C_current)

        S_current_list = S_current[S_current.columns[0]].tolist()
        D_current_list = D_current.values.tolist()

        # Re-assigning cluster label for each data point
        print('Re-assigning cluster labels\n')
        for i in range(len(L)):
            if L[i] not in S_current_list:
                temp = list(D_current_list[L[i]])
                temp2 = temp.copy()
                temp.sort(reverse=True)
                lenght = len(temp)
                while(lenght > 0):
                    min_d = temp.pop()
                    lenght -= 1
                    index = temp2.index(min_d)
                    if index in S_current_list:
                        L[i] = index
                        break
                temp.clear()

        ## Re-computing labels to make the be in range of C_current.
        L_set = set(L)
        counter = 0
        for i in L_set:
            for j in range(len(L)):
                if L[j] == i:
                    L[j] = counter
            counter += 1

        
        ## Re-computing D_current
        D_current = np.zeros((C_current, C_current))
        D_org_list = D_original.values.tolist()
        R_k_list = R_k.values.tolist()

        # Computing "P"s. "P"s contains all data points in a certain cluster and all their neighbours.
        print('Computing Ps\n')
        P = []
        for i in set(L):
            P_i = []
            for idx in range(len(L)):
                if L[idx] == i:
                    P_i.append(idx)
                    P_i.extend(R_k_list[idx])
            P_i = list(set(P_i))
            P.append(P_i)
        # Computing new D_current values
        print('Computing new D_current values\n')
        for i in range(C_current):
            for j in range(i, C_current):
                sum_val = 0
                P_i = P[i]
                P_j = P[j]
                len_P_i = len(P_i)
                len_P_j = len(P_j)
                for h in range(len(P_i)):
                    for q in range(len(P_j)):
                        sum_val += D_org_list[P_i[h]][P_j[q]]
                dist = sum_val/(len_P_i*len_P_j)
                D_current[i][j] = dist
                D_current[j][i] = dist
        D_current = pd.DataFrame(D_current)

        # Re-computing other parameters
        C_previous = C_current
        C_current = math.floor(C_current/G)
        

    ##7
    # Cease condition satisfied.
    print('Cease condition satisfied.\nIdentifying key elements for the last time.\n')
    S_current = identify_key_elements(D_current, C_target)

    S_current_list = S_current[S_current.columns[0]].tolist()
    D_current_list = D_current.values.tolist()


    # Re-assigning cluster label for each data point
    print('Re-assigning cluster labels\n')
    for i in range(len(L)):
        if L[i] not in S_current_list:
            temp = list(D_current_list[L[i]])
            temp2 = temp.copy()
            temp.sort(reverse=True)
            lenght = len(temp)
            while(lenght > 0):
                min_d = temp.pop()
                lenght -= 1
                index = temp2.index(min_d)
                if index in S_current_list:
                    L[i] = index
                    break
            temp.clear()


    ## Re-computing labels to make the be in range of C_current.
    L_set = set(L)
    counter = 0
    for i in L_set:
        for j in range(len(L)):
            if L[j] == i:
                L[j] = counter
        counter += 1

    print('Done\n')
    return pd.DataFrame(L, columns=["Clusters"])


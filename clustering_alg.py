import pandas as pd
import math
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.spatial import distance_matrix

### READING DATA

# Reading data from a csv file.
# The dataset contains fifa2017 player's stats
df = pd.read_csv('https://raw.githubusercontent.com/navinniish/Datasets/master/fifa.csv')

### NORMALIZING DATA

# Keeping records which have 'Overall' > 74 and discarding others. 
# In order to reduce the size of the data.
df = df[df['Overall'] > 74]

# Drop the records with unknown 'position' property.
# Position works as the label of the data for us, so unknown(NaN) position is not welcome.
df.dropna(subset=['Position'], axis=0, inplace=True)

# Extracting a subset of columns. Choosing only player's stats and skill points.
selected_columns = ['Crossing', 'Finishing', 'HeadingAccuracy',
       'ShortPassing', 'Volleys', 'Dribbling', 'Curve', 'FKAccuracy',
       'LongPassing', 'BallControl', 'Acceleration', 'SprintSpeed',
       'Agility', 'Reactions', 'Balance', 'ShotPower', 'Jumping',
       'Stamina', 'Strength', 'LongShots', 'Aggression', 'Interceptions',
       'Positioning', 'Vision', 'Penalties', 'Composure', 'Marking',
       'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling',
       'GKKicking', 'GKPositioning', 'GKReflexes']
label = df['Position']
# rdf: reduced df
rdf = df[selected_columns]

# Mapping every unique poisition in the dataset to 4 generaly known 
# and most common positions(Forward, Midfielder, Defence, Goalkeeper).
pos_map = {
    "F": ['CF', 'LF', 'RF', 'LS', 'ST', 'LW', 'RS', 'RW'],
    "M": ['CAM', 'CDM', 'CM', 'LAM', 'LCM', 'LDM', 'LM', 'RAM', 'RCM', 'RDM', 'RM'],
    "D": ['CB', 'LB', 'LCB', 'LWB', 'RB', 'RCB', 'RWB'],
    "G": ['GK']
    }
label = label.tolist()
for i in range(len(label)):
    mapped = False
    for j in pos_map.keys():
        if label[i] in pos_map[j]:
            label[i] = j
            mapped = True
            break
    if not mapped:
        print("Not considered this position. " + label[i] + " index: " + str(i) + ".")
label = pd.DataFrame(label,columns=['Position'])

### END OF NORMALIZING.

### SPLITING DATA.

#spliting data to train and test sets.
X_train, X_test, Y_train, Y_test = train_test_split(rdf, label, test_size=0.3, random_state=30)

X_train = pd.DataFrame(X_train, columns=selected_columns)
X_test = pd.DataFrame(X_test, columns=selected_columns)
Y_train = pd.DataFrame(Y_train, columns=['Position'])
Y_test = pd.DataFrame(Y_test, columns=['Position'])
print("X_train: " + str(X_train.shape))
print("X_test: " + str(X_test.shape))
print("Y_train: " + str(Y_train.shape))
print("Y_test: " + str(Y_test.shape))

### END OF SPLITING

### CLUSTRING ALGORITHM

# Identify key element.
# This function is called in each iteration to identify key elements of the new clusters.
def identify_key_elements(d_current: pd.DataFrame, C_current: int)->pd.DataFrame:
    pass


# This is the main routine of the algorithm.
def custom_clustering(X_train: pd.DataFrame)-> pd.DataFrame:

    # Decleration of the algorithm's parameters.
    DESIRED_CLUSTERS = 4
    CONSTANT_NUMBER_G = 2
    CONSTATN_NUMBER_K = 5

    ## 1
    # Computing distance matrix(D_original).
    D_original = pd.DataFrame(distance_matrix(X_train.values,X_train.values), index=list(range(X_train.shape[0])), columns=list(range(X_train.shape[0])))

    ##2
    # Finding K-Nearest data points.
    K = CONSTATN_NUMBER_K
    R_k = []
    for idx ,x in D_original.iterrows():
        R_k.append(list(x.nsmallest(K + 1).keys()))
    R_k = pd.DataFrame(R_k)

    ##3
    # Initializing cluster labels.
    N = X_train.shape[0]
    L = list(range(0,N))
    G = CONSTANT_NUMBER_G

    ##4
    # Initializing algorithm params.
    C_previous = N
    C_current = math.floor(N/G)
    C_target = DESIRED_CLUSTERS

    ##5
    # Initial Computation of D_current 
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

    ##6
    # Main loop of the algorithm

    while(C_current > C_target):
        S_current = identify_key_elements(D_current, C_current)

        S_current_list = S_current[S_current.columns[0]].tolist()
        D_current_list = D_current.values.tolist()

        # Re-assigning cluster label for each data point
        for i in range(len(L)):
            if i not in S_current_list:
                # min = math.inf
                # for j in D_current_list[i]:
                #     if j in S_current_list and j < min:
                #         min = j
                temp = list(D_current_list[i])
                temp.sort(reverse=True)
                while(len(temp)>0):
                    min_d = temp.pop()
                    if min_d in S_current_list:
                        L[i] = min_d
                        break
                temp.clear()

        ## Re-computing D_current
        D_current = np.zeros((C_current, C_current))
        D_org_list = D_original.values.tolist()
        R_k_list = R_k.values.tolist()

        # Computing "P"s. "P"s contains all data points in a certain cluster and all their neighbours.
        P = []
        for i in range(C_current):
            P_i = []
            for idx in range(len(L)):
                if L[idx] == i:
                    P_i.append(idx)
                    P_i.extend(R_k_list[idx])
            P_i = list(set(P_i))
            P.append(P_i)
        
        # Computing new D_current values
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
    S_current = identify_key_elements(D_current, C_current)

    S_current_list = S_current[S_current.columns[0]].tolist()
    D_current_list = D_current.values.tolist()

    # Re-assigning cluster label for each data point
    for i in range(len(L)):
        if i not in S_current_list:
            temp = list(D_current_list[i])
            temp.sort(reverse=True)
            while(len(temp)>0):
                min_d = temp.pop()
                if min_d in S_current_list:
                    L[i] = min_d
                    break
            temp.clear()

    return pd.DataFrame(L)
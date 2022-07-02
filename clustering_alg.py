import pandas as pd
from sklearn.model_selection import train_test_split
from clustering_tools import custom_clustering


### READING DATA

# Reading data from a csv file.
# The dataset contains fifa2017 player's stats
df = pd.read_csv('fifa.csv')

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

# This is the main routine of the program.
def main():
    print("Strating...")
    clusters = custom_clustering(X_train, 4)
    print(clusters)


if __name__ == "__main__":
    main()
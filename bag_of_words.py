from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import neural_network
import pandas as pd
import numpy as np
import re
from sklearn import metrics
from sklearn.metrics import adjusted_rand_score

df_train = pd.read_csv('./normalized_train.csv')
df_test = pd.read_csv('./normalized_test.csv') 
X_train = pd.DataFrame(df_train['Comment'])
X_test = pd.DataFrame(df_test['Comment'])
Y_train = pd.DataFrame(df_train['Topic'])
Y_test = pd.DataFrame(df_test['Topic'])

print("X_train: " + str(X_train.shape))
print("X_test: " + str(X_test.shape))
print("Y_train: " + str(Y_train.shape))
print("Y_test: " + str(Y_test.shape))

train_docs = X_train['Comment'].tolist()
test_docs = X_test['Comment'].tolist()

for i in range(len(train_docs)):
    train_docs[i] = re.sub('[.,#$@!?"()]', '', str(train_docs[i]))

for i in range(len(test_docs)):
    test_docs[i] = re.sub('[.,#$@!?"()]', '', str(test_docs[i]))

vectorizer = TfidfVectorizer(max_df=0.99, min_df=0.01)
train_Tfidf = vectorizer.fit_transform(train_docs).toarray()

feature_names = vectorizer.get_feature_names()

vocab = {}
for value, key in enumerate(feature_names):
    vocab[key] = value

vectorizer = TfidfVectorizer(vocabulary=vocab)
test_Tfidf = vectorizer.fit_transform(test_docs).toarray()

mlp = neural_network.MLPClassifier(hidden_layer_sizes=(128, 128, 32), learning_rate_init=0.001, max_iter=1000, tol=0.00001)
mlp.fit(train_Tfidf, np.array(Y_train['Topic'].tolist()))

predicted_labels = mlp.predict(test_Tfidf)

def purity_score(y_true, y_pred):
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

randIndex = adjusted_rand_score(predicted_labels, np.array(Y_test['Topic'].tolist()))
print('rand index: ', randIndex)
purity = purity_score(np.array(Y_test['Topic'].tolist()), predicted_labels)
print('purity: ', purity)
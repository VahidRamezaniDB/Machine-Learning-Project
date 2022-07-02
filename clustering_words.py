from clustering_tools import custom_clustering
from nltk.tokenize import sent_tokenize, word_tokenize
import pandas as pd
import gensim


df_train = pd.read_csv('./train.csv')
df_test = pd.read_csv('./test.csv')

print("Read data successfully")

comments_list = df_train['Comment'].tolist()
w2v_data = []
for comment in comments_list:
    for sentence in sent_tokenize(comment):
        temp = []
        for word in word_tokenize(sentence):
            temp.append(word.lower())
        w2v_data.append(temp)    

print("training W2V model")

model = gensim.models.Word2Vec(w2v_data, min_count = 1, vector_size = 30, window = 5, hs=1)

print(model.wv.similarity('cesium','observations'),model.wv.similarity('ride','car'))

words_vecs = pd.DataFrame(model.wv.vectors)
reduced_word_vecs = words_vecs.iloc[:5000]

print(reduced_word_vecs.shape)

words_labels = custom_clustering(reduced_word_vecs, 3)

print(set(words_labels[words_labels.columns[0]].tolist()))
from clustering_tools import custom_clustering
import pandas as pd
import pickle

print("Loading model")

model = pickle.load(open("w2v.sav", "rb"))

words_vecs = pd.DataFrame(model.wv.vectors)
reduced_word_vecs = words_vecs.iloc[:3000]

print(reduced_word_vecs.shape)
print("Clustring start...")
words_labels = custom_clustering(reduced_word_vecs, 3)

print(set(words_labels["Clusters"].tolist()))
print(words_labels["Clusters"].tolist()[:100])

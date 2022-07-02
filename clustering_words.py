import re
from clustering_tools import custom_clustering
import pandas as pd
import pickle

print("Loading model")

model = pickle.load(open("w2v.sav", "rb"))

my_dict = {}
for i in range(len(model.wv)):
    word = model.wv.index_to_key[i]
    my_dict[word] = model.wv[word]

reduced_word_vecs_list = list(my_dict.values())[:1000]

reduced_word_vecs = pd.DataFrame(reduced_word_vecs_list)

print(reduced_word_vecs.shape)
print("Clustring start...")
cluster_count = 20
words_labels = custom_clustering(reduced_word_vecs, cluster_count,5,2)

words_lab_list = words_labels["Clusters"].tolist()

print("Clustring finnished.\n Starting replacement.")

word_rep = []
word_list = list(my_dict.keys())
for i in range(cluster_count):
    for j in range(len(words_lab_list)):
        if words_lab_list[j] == i:
            word_rep.append(word_list[j])
            break

if len(word_rep) < cluster_count:
    print("ERROR!")
    exit(0)

print(word_rep)

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

print("Read files successfully.\n Replacing words.")

tr_comments = train_df["Comment"].tolist()
ts_comments = test_df["Comment"].tolist()
tr_topic = train_df["Topic"].tolist()
ts_topic = test_df["Topic"].tolist()

for idx in range(len(reduced_word_vecs_list)):
    for i in range(len(tr_comments)):
        regstr = model.wv.index_to_key[idx]
        tr_comments[i] = re.sub(r"\b%s\b" % regstr , word_rep[words_lab_list[idx]], tr_comments[i])
    for i in range(len(ts_comments)):
        regstr = model.wv.index_to_key[idx]
        ts_comments[i] = re.sub(r"\b%s\b" % regstr , word_rep[words_lab_list[idx]], ts_comments[i])

print("replaced words. Writing output.")
out_df = pd.DataFrame([tr_comments,tr_topic], index=["Comment", "Topic"]).T.to_csv("normalized_train.csv")
out_df = pd.DataFrame([ts_comments,ts_topic], index=["Comment", "Topic"]).T.to_csv("normalized_test.csv")
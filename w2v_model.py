from nltk.tokenize import sent_tokenize, word_tokenize
import pandas as pd
import gensim
import pickle
from nltk.corpus import stopwords


df_train = pd.read_csv('./train.csv')
df_test = pd.read_csv('./test.csv')

print("Read data successfully")

comments_list = df_train['Comment'].tolist()

ignore_these = ['(',')','[',']','{','}','\\','/','*','+','-','&',
                '^','>','<','"',"'","”",'’',':',';','$','“','#',
                ',','.','!','@','?','`','%']
for i in range(len(comments_list)):
    for j in ignore_these:
        comments_list[i] = comments_list[i].replace(j,'')

w2v_data = []
for comment in comments_list:
    for sentence in sent_tokenize(comment):
        temp = []
        for word in word_tokenize(sentence):
            temp.append(word.lower())
        w2v_data.append(temp)    

for i in w2v_data:
    temp = i.copy()
    for j in temp:
        if len(j) < 3:
            i.remove(j)
            
print("Removing stopwords")

stop_words = set(stopwords.words('english'))
for i in w2v_data:
    temp = i.copy()
    for j in temp:
        if j in stop_words:
            i.remove(j)

print("training W2V model")

model = gensim.models.Word2Vec(w2v_data, min_count = 1, vector_size = 30, window = 5, hs=1)

print(model.wv.similarity('cesium','observations'),model.wv.similarity('ride','car'))

pickle.dump(model, open("w2v.sav", "wb"))

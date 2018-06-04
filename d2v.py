import LabeledLineSentence
from gensim import *

train_sources = {"data/Doc2vec/train_1": "train_1",
           "data/Doc2vec/train_2": "train_2",
           "data/Doc2vec/train_3": "train_3",
           "data/Doc2vec/train_4": "train_4",
           "data/Doc2vec/train_5": "train_5",
            "data/Doc2vec/test_1": "test_1",
            "data/Doc2vec/test_2": "test_2",
            "data/Doc2vec/test_3": "test_3",
            "data/Doc2vec/test_4": "test_4",
            "data/Doc2vec/test_5": "test_5"}

train_reviews = LabeledLineSentence.LabeledLineSentence(train_sources)
n = len(train_reviews.to_array())
train_arr = train_reviews.to_array()
path = "data/Doc2vec/models/new/"

i=100
for j in [1,2,3]:
    model = models.doc2vec.Doc2Vec(size=i, window=j, min_count=1, workers=8)
    print("building model with size %d and window %d" % (i,j))
    model.build_vocab(train_arr)
    model.train(train_arr, total_examples=n, epochs=10)
    model.save(path + str(i) + '_' + str(j))







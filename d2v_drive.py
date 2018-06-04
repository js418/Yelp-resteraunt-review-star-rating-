import numpy
from gensim.models import Doc2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import LabeledLineSentence
from sklearn import svm

train_counts = [16364,13227,19465,36711,47431]
tr_labels = ["train_1","train_2","train_3","train_4","train_5"]
tn = sum(train_counts)

test_counts = [4313, 3991, 5697, 11566, 15658]
te_labels = ["test_1", "test_2", "test_3", "test_4", "test_5"]
test_n = sum(test_counts)

test_sources = {"data/Doc2vec/test_1":"test_1",
           "data/Doc2vec/test_2":"test_2",
           "data/Doc2vec/test_3":"test_3",
           "data/Doc2vec/test_4":"test_4",
           "data/Doc2vec/test_5":"test_5"}

test_reviews = LabeledLineSentence.LabeledLineSentence(test_sources)
test_list = test_reviews.to_array()

def test(model,size):
    train_arrays = numpy.zeros((tn, size))
    train_labels = numpy.zeros(tn)

    for k in range(5):
        for i in range(train_counts[k]):
            prefix_train = tr_labels[k] + '_' + str(i)
            train_arrays[i] = model.docvecs[prefix_train]
            train_labels[i] = k + 1

    test_arrays = numpy.zeros((test_n, size))
    test_labels = numpy.zeros(test_n)
    for i in range(test_n):
        test_arrays[i] = model.infer_vector(test_list[i][0])
        test_labels[i] = int(test_list[i][1][0][5])
    return train_arrays, train_labels,test_arrays, test_labels

def doc2vec_logReg(train_arrays, train_labels,test_arrays, test_labels):
    classifier = LogisticRegression()
    classifier.fit(train_arrays, train_labels)
    acc= classifier.score(test_arrays, test_labels)
    print("Accuracy: ")
    print(acc)
    return acc

def doc2vec_KNN(train_arrays, train_labels,test_arrays, test_labels,k):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_arrays, train_labels)
    pred_labels = knn.predict(test_arrays)
    count = 0.0
    for i in range(test_n):
        if (pred_labels[i] == test_labels[i]):
            count += 1.0
    acc = count / test_n
    print("Accuracy: ")
    print(acc)
    return acc

def doc2vec_SVM(train_arrays, train_labels,test_arrays, test_labels,kernel):
    svc = svm.SVC(kernel=kernel)
    svc.fit(train_arrays, train_labels)
    acc = svc.score(test_arrays, test_labels)
    print("Accuracy: ")
    print(acc)
    return acc

def main():
    size = 100
    model = Doc2Vec.load('data/Doc2vec/models/100_1')
    train_arrays, train_labels, test_arrays, test_labels = test(model, size)
    print("Doc2vec KNN:")
    for k in [3,5,7,9,10]:
        print("k = " +str(k))
        doc2vec_KNN(train_arrays, train_labels, test_arrays, test_labels, k)

    print("Doc2vec SVM:")
    for kernel in ['linear','poly','rbf']:
        print("kernel = " + kernel)
        doc2vec_SVM(train_arrays, train_labels, test_arrays, test_labels, kernel)

main()





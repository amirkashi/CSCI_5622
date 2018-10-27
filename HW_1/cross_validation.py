import argparse
import random
from collections import namedtuple
import numpy as np
from knn import Knearest, Numbers


random.seed(20170830)
SplitIndices = namedtuple("SplitIndices", ["train", "test"])


def split_cv(length, num_folds):
    splits = [SplitIndices([], []) for _ in range(num_folds)]
    indices = list(range(length))
    random.shuffle(indices)
    fold_lenght = int(length/num_folds)
    for i in range(0, len(splits)):
        first_ind_fold = fold_lenght * i
        last_ind_fold = first_ind_fold + fold_lenght
        test_index = list(indices[first_ind_fold:last_ind_fold])
        train_index = list(indices[0:len(indices)])
        for test_ind in test_index:
            train_index.remove(test_ind)
        for k in test_index:
            splits[i][1].append(k)
        for l in train_index:
            splits[i][0].append(l)
    return splits


def cv_performance(x, y, num_folds, k):
    length = len(y)
    splits = split_cv(length, num_folds)
    accuracy_array = []
    for split in splits:
        train_set_x = [x[i] for i in split.train]
        train_set_y = [y[i] for i in split.train]
        test_set_x = [x[i] for i in split.test]
        test_set_y = [y[i] for i in split.test]
        knn = Knearest(train_set_x, train_set_y, k)
        confusion_mtr = knn.confusion_matrix(test_set_x, test_set_y)
        accuracy = knn.accuracy(confusion_mtr)
        accuracy_array.append(accuracy)
    return np.mean(accuracy_array)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='KNN classifier options')
    parser.add_argument('--limit', type=int, default=-1,
                        help="Restrict training to this many examples")
    parser.add_argument('--folds', type=int, default=5,
                        help="get user defined fold numbers")
    args = parser.parse_args()
    data = Numbers("../data/mnist.pkl.gz")
    x, y = data.train_x, data.train_y
    if args.limit > 0:
        x, y = x[:args.limit], y[:args.limit]
    best_k, best_accuracy = -1, 0
    for k in [1, 3, 5, 7, 9]:
        if args.limit > 0:
            accuracy = cv_performance(x, y, args.limit, k)
        else:
            accuracy = cv_performance(x, y, 5, k)
        print("%d-nearest neighber accuracy: %f" % (k, accuracy))
        if accuracy > best_accuracy:
            best_accuracy, best_k = accuracy, k
    knn = Knearest(x, y, best_k)
    confusion = knn.confusion_matrix(data.test_x, data.test_y)
    accuracy = knn.accuracy(confusion)
    print("Accuracy for chosen best k= %d: %f" % (best_k, accuracy)
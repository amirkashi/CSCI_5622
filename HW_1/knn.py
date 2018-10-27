import argparse
import pickle
import gzip
from collections import Counter, defaultdict
import random
import numpy as np
from numpy import median
from sklearn.neighbors import BallTree


class Numbers:
    def __init__(self, location):
        with gzip.open(location, 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f)

        self.train_x, self.train_y = train_set
        self.test_x, self.test_y = valid_set
        f.close()


class Knearest:
    def __init__(self, x, y, k=5):
        self._kdtree = BallTree(x)
        self._y = y
        self._k = k

    def majority(self, item_indices):
        assert len(item_indices) == self._k, "Did not get k inputs"
        label_counts = {}
        for i in item_indices:
            if self._y[i] in label_counts:
                label_counts[self._y[i]] += 1
            else:
                label_counts[self._y[i]] = 1
        label_max = 0
        for i in label_counts:
            if label_max < label_counts[i]:
                label_max = label_counts[i]
        label_majority = []
        for labels in label_counts:
            if label_counts[labels] == label_max:
                label_majority.append(labels)
        if len(label_majority) == 1:
            return label_majority[0]
        else:
            Median = (median(np.array(label_majority)))
            return Median

    def classify(self, example):
        dist, ind = self._kdtree.query([example], k=self._k)
        return self.majority(ind[0])

    def confusion_matrix(self, test_x, test_y, debug=False):
        d = defaultdict(dict)
        data_index = 0
        for xx, yy in zip(test_x, test_y):
            classified = self.classify(xx)
            if classified in d[yy]:
                d[yy][classified] += 1
            else:
                d[yy][classified] = 1
            data_index += 1
        return d

    @staticmethod
    def accuracy(confusion_matrix):
        total = 0
        correct = 0
        for ii in confusion_matrix:
            total += sum(confusion_matrix[ii].values())
            correct += confusion_matrix[ii].get(ii, 0)
        if total > 0:
            return float(correct) / float(total)
        else:
            return 0.0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='KNN classifier options')
    parser.add_argument('--k', type=int, default=3,
                        help="Number of nearest points to use")
    parser.add_argument('--limit', type=int, default=-1,
                        help="Restrict training to this many examples")
    args = parser.parse_args()
    data = Numbers("../data/mnist.pkl.gz")
    if args.limit > 0:
        print("Data limit: %i" % args.limit)
        knn = Knearest(data.train_x[:args.limit], data.train_y[:args.limit],
                       args.k)
    else:
        knn = Knearest(data.train_x, data.train_y, args.k)
    print("Done loading data")
    confusion = knn.confusion_matrix(data.test_x, data.test_y)
    print("\t" + "\t".join(str(x) for x in range(10)))
    print("".join(["-"] * 90))
    for ii in range(10):
        print("%i:\t" % ii + "\t".join(str(confusion[ii].get(x, 0))
                                       for x in range(10)))
    print("Accuracy: %f" % knn.accuracy(confusion))

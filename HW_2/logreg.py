import random
import argparse
import gzip
import pickle
import numpy as np
from math import exp, log
from collections import defaultdict
SEED = 1735
random.seed(SEED)


class Numbers:
    def __init__(self, location):
        with gzip.open(location, 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f)
        self.train_x, self.train_y = train_set
        train_indices = np.where(self.train_y > 7)
        self.train_x, self.train_y = self.train_x[train_indices], 
                                     self.train_y[train_indices]
        self.train_y = self.train_y - 8
        self.valid_x, self.valid_y = valid_set
        valid_indices = np.where(self.valid_y > 7)
        self.valid_x, self.valid_y = self.valid_x[valid_indices], 
                                     self.valid_y[valid_indices]
        self.valid_y = self.valid_y - 8
        self.test_x, self.test_y = test_set
        test_indices = np.where(self.test_y > 7)
        self.test_x, self.test_y = self.test_x[test_indices], 
                                   self.test_y[test_indices]
        self.test_y = self.test_y - 8
    
    @staticmethod
    def shuffle(X, y):
        shuffled_indices = np.random.permutation(len(y))
        return X[shuffled_indices], y[shuffled_indices]


class LogReg:
    def __init__(self, num_features, eta):
        self.w = np.zeros(num_features)
        self.eta = eta
        self.last_update = defaultdict(int)

    def progress(self, examples_x, examples_y):
        logprob = 0.0
        num_right = 0
        for x_i, y in zip(examples_x, examples_y):
            p = sigmoid(self.w.dot(x_i))
            if y == 1:
                logprob += log(p)
            else:
                logprob += log(1.0 - p)
            if abs(y - p) < 0.5:
                num_right += 1
        return logprob, float(num_right) / float(len(examples_y))

    def sgd_update(self, x_i, y):
        y_estimated = sigmoid(self.w.dot(x_i))
        gradient_descent = -((y - y_estimated) * x_i)
        self.w = self.w - (self.eta * gradient_descent)
        return self.w

def sigmoid(score, threshold=20.0):
    if abs(score) > threshold:
        score = threshold * np.sign(score)
    sigmoid = 1.0 / (1.0 + exp(-(score)))
    return sigmoid

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--eta", help="Initial SGD learning rate",
                           type=float, default=0.1, required=False)
    argparser.add_argument("--passes", help="Number of passes through training data",
                           type=int, default=1, required=False)
    args = argparser.parse_args()
    data = Numbers('../data/mnist.pkl.gz')
    lr = LogReg(data.train_x.shape[1], args.eta)
    number_of_features = data.train_x.shape[1]
    iteration = 0
    for epoch in range(args.passes):
        data.train_x, data.train_y = Numbers.shuffle(data.train_x, data.train_y)
        for feature in range(0, number_of_features):
            X = data.train_x[feature]
            Y = data.train_y[feature]
            lr.sgd_update(X, Y)
        training_log_prob, training_accuracy = lr.progress(data.train_x, data.train_y)
        test_log_prob, test_accuracy = lr.progress(data.test_x, data.test_y)
        print ("Accuracy for ", epoch, " = ", round(training_accuracy, 2), 
               " for training data set and = ", round(test_accuracy, 2), " for test dataset")

import argparse
import pickle
import gzip
from collections import Counter, defaultdict
import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Activation
from keras.layers.core import Reshape
from keras.utils import to_categorical
from keras import backend as K


class Numbers:
    def __init__(self, location):
        with gzip.open(location, 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f)
        self.train_x, self.train_y = train_set
        self.test_x, self.test_y = valid_set

class CNN:
    def __init__(self, train_x, train_y, test_x, test_y, epoches=10, batch_size=128):
        self.batch_size = batch_size
        self.epoches = epoches
        n = 1
        width = 28
        height = 28
        self.train_x = train_x.reshape(train_x.shape[0], width, height, n)
        self.test_x = test_x.reshape(test_x.shape[0], width, height, n)
        self.train_x = self.train_x.astype('float32')
        self.test_x = self.test_x.astype('float32')
        self.train_y = to_categorical(train_y)
        self.test_y = to_categorical(test_y)
        self.model = Sequential()
        self.model.add(Conv2D(32, 5, 5, activation='relu', input_shape=(28, 28, 1)))
        self.model.add(Conv2D(32, 5, 5, activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(10, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', 
        					metrics=['accuracy'])

    def train(self):
        self.model.fit(self.train_x, self.train_y, batch_size=self.batch_size
        				, nb_epoch=self.epoches, verbose=1)

    def evaluate(self):
        acc = self.model.evaluate(self.test_x, self.test_y)
        return acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CNN classifier options')
    parser.add_argument('--limit', type=int, default=-1,
                        help='Restrict training to this many examples')
    args = parser.parse_args()
    data = Numbers("../data/mnist.pkl.gz")
    cnn = CNN(data.train_x[:args.limit], data.train_y[:args.limit], data.test_x, data.test_y)
    cnn.train()
    acc = cnn.evaluate()
    print(acc)
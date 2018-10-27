import argparse
import numpy as np
import pickle, gzip
import matplotlib.pyplot as plt


class Network:
    def __init__(self, sizes):
        self.L = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(n, 1) for n in self.sizes[1:]]
        self.weights = [np.random.randn(n, m) for (m, n) in zip(self.sizes[:-1], self.sizes[1:])]        

    def g(self, z):
        return sigmoid(z)

    def g_prime(self, z):
        return sigmoid_prime(z)

    def forward_prop(self, a):
        for (W, b) in zip(self.weights, self.biases):
            a = self.g(np.dot(W, a) + b)
        return a

    def gradC(self, a, y):
        return (a - y)

    def SGD_train(self, train, epochs, eta, lam=0.0, verbose=True, test=None):
        n_train = len(train)
        for epoch in range(epochs):
            perm = np.random.permutation(n_train)
            for kk in range(n_train):
                xk = train[perm[kk]][0]
                yk = train[perm[kk]][1]
                dWs, dbs = self.back_prop(xk, yk)
                self.weights = [W - eta*dW - eta*lam*W for (W, dW) in zip(self.weights, dWs)]
                self.biases = [b - eta*db for (b, db) in zip(self.biases, dbs)]
            if verbose:
                if epoch == 0 or (epoch + 1) % 15 == 0:
                    acc_train = self.evaluate(train)
                    if test is not None:
                        acc_test = self.evaluate(test)
                        print("Epoch {:4d}: Train {:10.5f}, Test {:10.5f}".format(epoch+1, acc_train, acc_test))
                    else:
                        print("Epoch {:4d}: Train {:10.5f}".format(epoch+1, acc_train))

    def back_prop(self, x, y):

        db_list = [np.zeros(b.shape) for b in self.biases]
        dW_list = [np.zeros(W.shape) for W in self.weights]
        a = x
        a_list = [a]
        z_list = [np.zeros(a.shape)]
        for W, b in zip(self.weights, self.biases):
            z = np.dot(W, a) + b
            z_list.append(z)
            a = self.g(z)
            a_list.append(a)
        gradient = (y - a_list[self.L-1])
        g_prime_z = (self.g_prime(z_list[self.L-1]))
        delta = -1.0 * (gradient * g_prime_z)
        y = np.squeeze(delta, axis=1)
        for ell in range(self.L-2, -1, -1):
            db_list[ell] = np.squeeze(delta, axis=1)
            db_list[ell] = np.expand_dims(db_list[ell], axis=1)
            a_list_transpose = np.transpose(a_list[ell])
            dW_list[ell] = (np.dot(delta, a_list_transpose))
            gadient = np.dot(np.transpose(self.weights[ell]), delta)
            g_prime_z = self.g_prime(z_list[ell])
            delta = gadient * g_prime_z
        return (dW_list, db_list)

    def evaluate(self, test):
        ctr = 0
        for x, y in test:
            yhat = self.forward_prop(x)
            ctr += np.argmax(yhat) == np.argmax(y)
        return float(ctr) / float(len(test))

    def compute_cost(self, x, y):
        a = self.forward_prop(x)
        return 0.5 * np.linalg.norm(a - y) ** 2

def sigmoid(z, threshold=20):
    z = np.clip(z, -threshold, threshold)
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1.0 - sigmoid(z))

def mnist_digit_show(flatimage, outname=None):
    import matplotlib.pyplot as plt
    image = np.reshape(flatimage, (-1, 14))
    plt.matshow(image, cmap=plt.cm.binary)
    plt.xticks([])
    plt.yticks([])
    if outname:
        plt.savefig(outname)
    else:
        plt.show()

if __name__ == "__main__":
    f = gzip.open('../data/tinyMNIST.pkl.gz', 'rb')
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    train, test = u.load()
    nn = Network([196, 200, 10])
    nn.SGD_train(train, epochs=500, eta=0.25, lam=0.0, verbose=True, test=test)

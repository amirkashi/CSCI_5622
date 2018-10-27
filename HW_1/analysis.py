from knn import Knearest, Numbers
import matplotlib.pyplot as plt
from collections import Counter, defaultdict


def limit(lim, tr_x, tr_y, ts_x, ts_y, K):
	accuracy = {}
	plt.ion()
	for k in K:
		accr = []
		for l in lim:
			knn = Knearest(tr_x[:l], tr_y[:l], k)
			conf = knn.confusion_matrix(ts_x[:l], ts_y[:l])
			ac = knn.accuracy(conf)
			accr.append(ac)
		accuracy[k] = accr
	plt.ion()
	for vals in accuracy:
		plt.plot(lim, accuracy[vals], label="K= " + str(vals))
		plt.xlabel("Numbers of Training")
		plt.ylabel("Accuracy")
		plt.title("Figure 1 - Accuracies Against Numbers of Training ")
		plt.legend(bbox_to_anchor=(0.5, 0.5), loc=2, borderaxespad=0.)
		plt.savefig("question_1.png")
		plt.show()

if __name__ == "__main__":
	data = Numbers("../data/mnist.pkl.gz")
	train_x, train_y = data.train_x, data.train_y
	test_x, test_y = data.test_x, data.test_y
	k = list(range(1, 10, 2))
	limits = list(range(100, 2001, 100))
	limit(limits, train_x, train_y, test_x , test_y, k)
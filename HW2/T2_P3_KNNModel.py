import numpy as np
from collections import Counter

# Please implement the predict() method of this class
# You can add additional private methods by beginning them with
# two underscores. It may look like the __dummyPrivateMethod below. You can feel
# free to change any of the class attributes, as long as you do not change any
# of the given function headers (they must take and return the same arguments).

class KNNModel:
    def __init__(self, k):
        self.X = None
        self.y = None
        self.K = k

    # Find plurality class of the k nearest neighbors
    def __majority(self, nearest_ys):
        count = Counter(nearest_ys)
        return count.most_common(1)[0][0]

    def predict(self, X_pred):
        # Calculate all distances to training data points
        dists = np.empty([X_pred.shape[0], 0])
        for x_train in self.X:
            diff = X_pred - x_train
            dist = ((diff[:,0] / 3) ** 2 + diff[:,1] ** 2).reshape(-1, 1)
            dists = np.hstack((dists, dist))
        # Find KNN and predict based on their plurality class
        preds = []
        for dist in dists:
            dist_y = [(d, self.y[i]) for i, d in enumerate(dist)]
            dist_y.sort(key = lambda t: t[0])
            nearest_ys = [y_train for _, y_train in dist_y[0:self.K]]
            pred = self.__majority(nearest_ys)
            preds.append(pred)
        return np.array(preds)

    # In KNN, "fitting" can be as simple as storing the data, so this has been written for you
    # If you'd like to add some preprocessing here without changing the inputs, feel free,
    # but it is completely optional.
    def fit(self, X, y):
        self.X = X
        self.y = y

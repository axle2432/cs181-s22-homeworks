import numpy as np
from scipy.stats import multivariate_normal as mvn  # you may find this useful

# Please implement the fit(), predict(), and negative_log_likelihood() methods
# of this class. You can add additional private methods by beginning them with
# two underscores. It may look like the __dummyPrivateMethod below. You can feel
# free to change any of the class attributes, as long as you do not change any
# of the given function headers (they must take and return the same arguments).

class GaussianGenerativeModel:
    def __init__(self, is_shared_covariance=False):
        self.is_shared_covariance = is_shared_covariance
        self.k = 3 # Number of classes
        self.pi = None # Prior for each class, k * 1
        self.mvns = [] # MVN for each class with MLE parameters

    # Probability of x belonging to class j (times p(x), same for all classes)
    def __prob(self, x, j):
        return self.mvns[j].pdf(x) * self.pi[j]

    def fit(self, X, y):
        # Convert to one-hot representation, n * k
        y = np.array([[int(y_i == j) for j in range(self.k)] for y_i in y])
        d = X.shape[1]

        # Estimate parameters from training data
        self.pi = np.sum(y, axis=0).T / np.sum(y) # k * 1
        mu = (y.T @ X) / np.sum(y, axis=0).T.reshape(-1, 1) # k * d
        if self.is_shared_covariance:
            sigma_shared = np.zeros([d, d])
        else:
            sigma_separate = []
        for j in range(self.k):
            class_diffs = (X - mu[j])[y[:,j] == 1]
            class_cov_n = class_diffs.T @ class_diffs
            if self.is_shared_covariance:
                sigma_shared += class_cov_n
            else:
                class_cov = class_cov_n / np.sum(y[:,j])
                sigma_separate.append(class_cov)
        if self.is_shared_covariance:
            sigma_shared /= np.sum(y)
        for j in range(self.k):
            s = sigma_shared if self.is_shared_covariance else sigma_separate[j]
            self.mvns.append(mvn(mean=mu[j], cov=s))

    def predict(self, X_pred):
        preds = []
        for x in X_pred:
            preds.append(np.argmax([self.__prob(x, j) for j in range(self.k)]))
        return np.array(preds)

    def negative_log_likelihood(self, X, y):
        return -sum([np.log(self.__prob(x, j)) for (x, j) in zip(X, y)])


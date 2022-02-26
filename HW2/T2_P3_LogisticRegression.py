import numpy as np
from scipy.special import softmax as softmax
import matplotlib.pyplot as plt
from tqdm import tqdm

# Please implement the fit(), predict(), and visualize_loss() methods of this
# class. You can add additional private methods by beginning them with two
# underscores. It may look like the __dummyPrivateMethod below. You can feel
# free to change any of the class attributes, as long as you do not change any
# of the given function headers (they must take and return the same arguments).

class LogisticRegression:
    def __init__(self, eta, lam):
        self.eta = eta
        self.lam = lam
        self.k = 3 # number of classes
        self.W = None # weights, d * k
        self.runs = 200000
        self.W_opt = None # save best training weights before visualize_loss
        self.X = None # save transformed training data for visualize_loss
        self.y = None
        self.iter_loss = None # list of (iter, loss) pairs for visualize_loss

    # Add bias term
    def __basis(self, X):
        return np.hstack([np.ones([X.shape[0], 1]), X])

    # Probability for each input of each class
    def __probs(self, X):
        Z = X @ self.W # n * k
        return softmax(Z, axis=1) # n * k

    # Negative log-likelihood loss
    def __loss(self, X, y):
        probs = self.__probs(X)
        return -sum([np.log(p_i[y_i == 1]) for (p_i, y_i) in zip(probs, y)])

    # Optimize using gradient descent
    def fit(self, X, y, visualize_loss=False):
        if not visualize_loss:
            # Add bias term to X and convert y to n * k one-hot representation
            X = self.__basis(X)
            y = np.array([[int(y_i == j) for j in range(self.k)] for y_i in y])
            self.X = X
            self.y = y
        else:
            X = self.X
            y = self.y

        self.W = np.random.rand(X.shape[1], self.k) # d * k
        if visualize_loss:
            self.iter_loss = [(0, self.__loss(X, y))]
        #for iteration in range(self.runs):
        for iteration in tqdm(range(self.runs)):
            # Compute gradients and update weights
            probs = self.__probs(X)
            for j in range(self.k):
                grad_j = X.T @ (probs[:,j] - y[:,j])
                grad_j += 2 * self.lam * self.W[:,j] # regularization
                self.W[:,j] = self.W[:,j] - self.eta * grad_j
            if visualize_loss:
                self.iter_loss.append((iteration, self.__loss(X, y)))

    def predict(self, X_pred):
        X_pred = self.__basis(X_pred)
        # Argmax converts back to non-one-hot representation
        return np.argmax(self.__probs(X_pred), axis=1) # n * 1

    def visualize_loss(self, output_file, show_charts=False):
        plt.figure()
        plt.title(output_file)
        plt.xlabel("Number of Iterations")
        plt.ylabel("Negative Log-Likelihood Loss")
        plt.yscale("log")
        self.W_opt = self.W # save best weights
        eta = 0.1
        lam = 0.1
        self.eta = eta
        self.lam = lam
        self.fit(self.X, self.y, visualize_loss=True)
        it, loss = zip(*self.iter_loss)
        plt.plot(it, loss, label=r'$\eta = {0}, \lambda = {1}$'.format(eta, lam))
        for eta in [0.05, 0.01, 0.001]:
            self.eta = eta
            for lam in [0.05, 0.01, 0.001]:
                self.lam = lam
                self.fit(self.X, self.y, visualize_loss=True)
                it, loss = zip(*self.iter_loss)
                plt.plot(it, loss, label=r'$\eta = {0}, \lambda = {1}$'.format(eta, lam))
        self.W = self.W_opt # restore best weights
        plt.legend()
        plt.savefig(output_file + '.png')
        if show_charts:
            plt.show()


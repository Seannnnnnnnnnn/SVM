from svm import *


class DualSVM(SVM):
    """Soft-margin SVM using dual formulation, training
    with stochastic gradient ascent

    Parameters
    ----------
    eta : float
        Learning rate.
    C: float
        Regularization parameter.
    kernel: Kernel
        Kernel function
    """

    def __init__(self, eta, C, kernel=None):
        super().__init__()
        if kernel is None:
            def dot_product(u, v):
                return np.dot(u, v.T)

            self.kernel = dot_product
        else:
            self.kernel = kernel

        # your code here
        self.eta = eta
        self.C = C
        self.X = None
        self.y = None
        self.alphas = None
        self.bias = None
        self.primal = None
        # end of your code

    def fit(self, X, y, iterations=100):

        super().fit(X=X, y=y)
        self.X = X
        self.y = y

        # your code here
        self.alphas = self.__train(X, y, iterations)

        # end of your code
        self.bias = self.get_bias()
        return

    def predict(self, test_X):
        super().predict(test_X=test_X)
        # your code here
        y_s = []
        for x_i in test_X:
            y_i = 1 if (self.bias + self.__compute_sub_gradient(self.X, x_i, self.y, self.alphas)) >= 0 else -1
            y_s.append(y_i)
        return np.array(y_s)
        # end of your code

    def primal_weights(self):
        """Compute weights based on alphas, assuming linear kernel
        """
        # your code here
        # your code here
        assert self.alphas is not None  # ensure this cannot be called if we have not computed the alphas

        if self.primal:
            return self.primal
        else:
            n, l = self.X.shape[1], self.X.shape[0]

            w = np.zeros(n)
            for i in range(n):
                for j in range(l):
                    w[i] += self.alphas[j] * self.y[j] * self.X[j][i]
            self.primal = w
            return self.primal
        # end of your code

    def get_bias(self):
        """Compute bias based on learned alphas and training data set
        """
        # your code here
        n = self.X.shape[1]
        biases = np.zeros(n)
        for i in range(n):
            biases[i] = self.y[i] - self.__compute_sub_gradient(self.X, self.X[i], self.y, self.alphas)
        return biases.mean()
        # end of your code

    def __train(self, X, y, n_iter):
        l, eta = X.shape[0], self.eta
        alpha = np.zeros(l)

        for _ in range(n_iter):
            for i in range(l):
                subgradient = self.__compute_sub_gradient(X, X[i], y, alpha)
                alpha[i] = alpha[i] + eta * (1 - y[i] * subgradient)

                if alpha[i] < 0:
                    alpha[i] = 0
                elif alpha[i] > self.C:
                    alpha[i] = self.C

        return alpha

    def __compute_sub_gradient(self, X, x_i, y, alpha):
        grad = 0
        for j in range(X.shape[0]):
            grad += alpha[j] * y[j] * self.kernel(x_i, X[j])
        return grad

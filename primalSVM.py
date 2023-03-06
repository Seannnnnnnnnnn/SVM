from svm import *


class PrimalSVM(SVM):
    """Soft-margin SVM fit using primal objective, training
    with stochastic gradient ascent.

    Parameters
    ----------
    eta : float
        Learning rate.
    lambda0: float
        Regularisation term, must be strictly positive
    """

    def __init__(self, eta, lambda0) -> None:
        super().__init__()
        self.eta = eta
        self.lambda0 = lambda0
        self.w = None
        self.b = None

    def fit(self, X, y, iterations=100) -> None:
        super().fit(X=X, y=y)
        w, b = np.zeros(X.shape[1]), 0
        self.w, self.b = self._train(X, y, w, b, iterations)

    def predict(self, test_X):
        super().predict(test_X=test_X)
        y_s = []
        for x_i in test_X:
            y_i = 1 if np.dot(self.w, x_i) + self.b >= 1 else -1
            y_s.append(y_i)

        return np.array(y_s)

    def _compute_sub_gradients(self, X, y, w, b):
        subgrad_w = 0
        subgrad_b = 0

        # sum over all sub-gradients of hinge loss for a given samples x,y
        for x_i, y_i in zip(X, y):
            f_xi = np.dot(w.T, x_i) + b

            decision_value = y_i * f_xi

            if decision_value < 1:
                subgrad_w += - y_i * x_i
                subgrad_b += -1 * y_i
            else:
                subgrad_w += 0
                subgrad_b += 0

        # multiply by C after summation of all sub-gradients for a given samples of x,y
        subgrad_w += self.lambda0
        subgrad_b += self.lambda0
        return w+subgrad_w, subgrad_b

    def _train(self, X, y, w, b, n_iter, tol=1e-6):
        # trains using SGD
        eta = self.eta
        for i in range(n_iter):

            sub_grads = self._compute_sub_gradients(X, y, w, b)
            w = w - eta * sub_grads[0]
            b = b - eta * sub_grads[1]
        return w, b

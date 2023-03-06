from svm import *


def random_pairs_heuristic(svm):
    for i2 in range(svm.n_samples):
        i1 = random.randint(0, svm.n_samples - 1)
        yield (i1, i2)


def all_pairs_heuristic(svm):
    for i2 in range(svm.n_samples):
        for i1 in range(i2 + 1, svm.n_samples):
            yield (i1, i2)


class SMO(SVM):
    def __init__(self, C, eps, tol, kernel, heuristic=all_pairs_heuristic):
        super().__init__()
        """Soft-margin SVM using dual formulation, training with sequential minimal optimization 
        """
        self.C = C
        self.eps = eps
        self.tol = tol
        self.kernel = kernel
        self.heuristic = heuristic

    def fit(self, X, y, iterations=100):

        """Fit function for training SVM

        Parameters
        ----------
        iterations: int
            Stop training after this many iterations, even if not converged.
        """

        super().fit(X=X, y=y)
        self.X = X
        self.y = y
        self.b = 0
        self.alphas = np.zeros(X.shape[0])
        self.n_samples, self.n_features = X.shape
        self.K = self.kernel(self.X, self.X)
        self.update_errors()

        num_changed = 1
        complete_pass = False
        epoch = 1
        total_changed = total_examined = 0
        for epoch in tqdm.trange(iterations):
            num_changed = num_examined = 0
            if not complete_pass:
                to_visit = self.heuristic(self)
            else:
                # force a full pass over all pairs of points
                # if previous epoch had no updates, to confirm
                # that model has converged
                to_visit = all_pairs_heuristic(self)
            for i1, i2 in to_visit:
                if self.__take_step(i1, i2):
                    num_changed += 1
                num_examined += 1

            epoch += 1
            total_changed += num_changed
            total_examined += num_examined
            if complete_pass and num_changed == 0:
                break
            complete_pass = (num_changed == 0)

        if complete_pass and num_changed == 0:
            print('converged, total changed', total_changed, 'examined', total_examined)
        else:
            print('not converged, total changed', total_changed, 'examined', total_examined)

        return

    def lower(self, i1, i2):
        # your code here ###
        if self.y[i1] != self.y[i2]:
            return max(0, self.alphas[i2] - self.alphas[i1])
        else:
            return max(0, self.alphas[i2] - self.alphas[i1]-self.C)
        # end of your code ###

    def higher(self, i1, i2):
        # your code here
        if self.y[i1] != self.y[i2]:
            return min(self.C, self.C+self.alphas[i2]-self.alphas[i1])
        else:
            return min(self.C, self.alphas[i2]+self.alphas[i1])
        # end of your code


    def update_errors(self):
        self.errors = np.dot(self.alphas * self.y, self.K) - self.b - self.y

    def __take_step(self, i1, i2):
        if (i1 == i2):
            return False

        # extract alphas for two instances and calculate errors
        alpha1 = self.alphas[i1]
        alpha2 = self.alphas[i2]
        y1 = self.y[i1]
        y2 = self.y[i2]
        E1 = self.errors[i1]
        E2 = self.errors[i2]
        s = y1 * y2

        L = self.lower(i1, i2)
        H = self.higher(i1, i2)

        if L == H:
            return False

        # your code here
        k11 = self.kernel(self.X[i1], self.X[i1])
        k12 = self.kernel(self.X[i1], self.X[i2])
        k22 = self.kernel(self.X[i2], self.X[i2])
        eta = k11 + k12 + k22
        if eta <= 0:
            return False  # we will ignore this corner case

        new_a2 = alpha2 + y2*(E1-E2)/eta
        if new_a2 < L: new_a2 = L
        elif new_a2 > H: new_a2 = H
        # end of your code

        if abs(new_a2 - alpha2) < self.eps * (new_a2 + alpha2 + self.eps):
            return False

        # update alphas, bias and errors
        new_a1 = alpha1 + s * (alpha2 - new_a2)
        self.b = self.update_bias(new_a1, new_a2, i1, i2, k11, k12, k22)
        self.update_errors()
        self.alphas[i1] = new_a1
        self.alphas[i2] = new_a2

        return True

    def update_bias(self, new_a1, new_a2, i1, i2, k11, k12, k22):

        # extract alphas for two instances and calculate errors
        alpha1 = self.alphas[i1]
        alpha2 = self.alphas[i2]
        y1 = self.y[i1]
        y2 = self.y[i2]
        E1 = self.errors[i1]
        E2 = self.errors[i2]

        b1 = E1 + y1 * (new_a1 - alpha1) * k11 + y2 * (new_a2 - alpha2) * k12 + self.b
        b2 = E2 + y1 * (new_a1 - alpha1) * k12 + y2 * (new_a2 - alpha2) * k22 + self.b

        if 0 < new_a1 < self.C:
            self.b = b1
        elif 0 < new_a2 < self.C:
            self.b = b2
        else:
            self.b = (b1 + b2) * 0.5


    def predict(self, test_X):
        super().predict(test_X=test_X)
        y_s = []
        for x_i in test_X:
            y_i = 1 if (-self.b + self.__compute_sub_gradient(self.X, x_i, self.y, self.alphas)) >= 0 else -1
            y_s.append(y_i)
        return np.array(y_s)

    def __compute_sub_gradient(self, X, x_i, y, alpha):
        grad = 0
        for j in range(X.shape[0]):
            grad += alpha[j] * y[j] * self.kernel(x_i, X[j])
        return grad

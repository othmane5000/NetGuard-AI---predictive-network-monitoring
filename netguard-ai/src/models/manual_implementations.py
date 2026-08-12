"""
NETGUARD AI - Manual ML Implementations (NumPy only)
=======================================================

Purpose: demonstrate a real understanding of the mechanics behind
scikit-learn's LogisticRegression / LinearRegression, as required by the
"Supervised Machine Learning: Regression and Classification" course
material - cost function, gradient descent, and L2 regularization written
from scratch, then benchmarked against sklearn in train_classification.py
and train_regression.py.
"""

import numpy as np


def sigmoid(z):
    z = np.clip(z, -500, 500)  # avoid overflow in exp
    return 1 / (1 + np.exp(-z))


class ManualLogisticRegression:
    """
    Binary logistic regression trained with batch gradient descent.

    Model:      p(y=1|x) = sigmoid(w . x + b)
    Cost:       J(w,b) = -1/m * sum[ y*log(p) + (1-y)*log(1-p) ]
                          + (lambda / 2m) * sum(w_j^2)      <- L2 regularization
                (binary cross-entropy / log loss, the standard cost
                 function for logistic regression; the second term is
                 the L2 / "ridge-style" regularization penalty which
                 shrinks weights to reduce overfitting; bias b is NOT
                 regularized, per convention.)
    Gradients:  dJ/dw = 1/m * X.T . (p - y)  + (lambda/m) * w
                dJ/db = 1/m * sum(p - y)
    """

    def __init__(self, learning_rate=0.1, n_iterations=2000, lambda_reg=0.0,
                 verbose=False, print_every=200):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.lambda_reg = lambda_reg
        self.verbose = verbose
        self.print_every = print_every
        self.w = None
        self.b = 0.0
        self.cost_history = []

    def _cost(self, X, y, w, b):
        m = X.shape[0]
        z = X @ w + b
        p = sigmoid(z)
        eps = 1e-12  # numerical stability for log(0)
        cross_entropy = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
        reg_term = (self.lambda_reg / (2 * m)) * np.sum(w ** 2)
        return cross_entropy + reg_term

    def fit(self, X, y):
        m, n = X.shape
        self.w = np.zeros(n)
        self.b = 0.0
        y = np.asarray(y, dtype=float)

        for i in range(self.n_iterations):
            z = X @ self.w + self.b
            p = sigmoid(z)
            error = p - y  # (m,)

            dw = (X.T @ error) / m + (self.lambda_reg / m) * self.w
            db = np.mean(error)

            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

            cost = self._cost(X, y, self.w, self.b)
            self.cost_history.append(cost)

            if self.verbose and i % self.print_every == 0:
                print(f"  iter {i:5d}  cost={cost:.5f}")

        return self

    def predict_proba(self, X):
        return sigmoid(X @ self.w + self.b)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


class ManualLinearRegression:
    """
    Multivariate linear regression trained with batch gradient descent.

    Model:      y_hat = w . x + b
    Cost:       J(w,b) = 1/2m * sum( (y_hat - y)^2 ) + (lambda/2m) * sum(w_j^2)
                (mean-squared-error cost function, the standard cost for
                 linear regression, with an L2 / ridge regularization term)
    Gradients:  dJ/dw = 1/m * X.T . (y_hat - y)  + (lambda/m) * w
                dJ/db = 1/m * sum(y_hat - y)
    """

    def __init__(self, learning_rate=0.1, n_iterations=2000, lambda_reg=0.0,
                 verbose=False, print_every=200):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.lambda_reg = lambda_reg
        self.verbose = verbose
        self.print_every = print_every
        self.w = None
        self.b = 0.0
        self.cost_history = []

    def _cost(self, X, y, w, b):
        m = X.shape[0]
        y_hat = X @ w + b
        mse = np.mean((y_hat - y) ** 2) / 2
        reg_term = (self.lambda_reg / (2 * m)) * np.sum(w ** 2)
        return mse + reg_term

    def fit(self, X, y):
        m, n = X.shape
        self.w = np.zeros(n)
        self.b = 0.0
        y = np.asarray(y, dtype=float)

        for i in range(self.n_iterations):
            y_hat = X @ self.w + self.b
            error = y_hat - y

            dw = (X.T @ error) / m + (self.lambda_reg / m) * self.w
            db = np.mean(error)

            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

            cost = self._cost(X, y, self.w, self.b)
            self.cost_history.append(cost)

            if self.verbose and i % self.print_every == 0:
                print(f"  iter {i:5d}  cost={cost:.5f}")

        return self

    def predict(self, X):
        return X @ self.w + self.b

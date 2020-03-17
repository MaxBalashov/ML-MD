import numpy as np
from numpy.random import seed
from sklearn.utils import resample
from sklearn.base import BaseEstimator, RegressorMixin


# LOSS and MAE functions
def mae(y_true, y_pred):
    return np.mean(np.absolute(y_true - y_pred))


def loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


class LinRegEstimator(BaseEstimator, RegressorMixin):
    '''
    Inherited methods:
    BaseEstimator - get_params, set_params
    RegressorMixin - score
    '''
    _estimator_type = 'regressor'


class LinearRegression(LinRegEstimator):
    def __init__(self, alpha=0.001, n_iter=1000, fit_intercept=True, batch_size=None, random_state=None):
        self.alpha = alpha
        self.n_iter = n_iter
        self.fit_intercept = fit_intercept
        self.batch_size = batch_size
        self.random_state = random_state
        self.w = None

    def _predict(self, X, w):
        return np.dot(X, w)
        
    def grad(self, X, y, w):
        N = len(y)
        return -(2 / N * X.T * (y - self._predict(X, w)))

    def gradient_descent(self, X, y, w, alpha=0.001, n_iter=1000,
                         batch_size=None, random_state=None):
        seed(seed=random_state)
        for _ in range(n_iter): # mini-batch (=const), stochastic (=1)
            if batch_size:
                X_batch, y_batch = resample(X, y, n_samples=batch_size, replace=False)
            else: # batch (=all)
                X_batch, y_batch = X, y
            w -= alpha * self.grad(X_batch, y_batch, w).sum(axis=1)
        return w

    def fit(self, X, y):
        # есть ли свободный член
        if self.fit_intercept:
            # добавим колонку из единиц
            X = np.c_[np.ones(len(X)), X]
        # инициализация весов линейной регрессии (нулями)
        w = np.zeros(X.shape[1])
        # градиентный спуск для подбора весов
        w = self.gradient_descent(X, y, w,
                                  alpha=self.alpha,
                                  n_iter=self.n_iter,
                                  batch_size=self.batch_size,
                                  random_state=self.random_state)
        # запись полученных весов в атрибут
        self.w = w

    def predict(self, X):
        # есть ли свободный член
        if self.fit_intercept:
            # добавим колонку из единиц
            X = np.c_[np.ones(len(X)), X]
        return self._predict(X, self.w)
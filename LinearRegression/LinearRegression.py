import numpy as np
from numpy.linalg import norm
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import resample

class LinRegEstimator(BaseEstimator, RegressorMixin):
    '''
    Inherited methods:
    BaseEstimator - get_params, set_params
    RegressorMixin - score
    '''
    _estimator_type = 'regressor'

class LinearRegression(LinRegEstimator):
    '''
    Linear regression optimized by gradient descent.
    ---------------------------------------------------------------------------
    Parameters:

    eta : float
        Learning rate for gradient step (~ 3e-9)
    epsilon : float (default 0.005)
        Stop criteria parametr
        Lowest value for difference between two latest sets of coefficients
        Difference computed as a norm (Frobenius)
        Can take following values:
            - None => eliminate stop criteria, all epochs will be computed
            - numeric (0.005)
    teta : float (default 0.005)
        Stop criteria parametr
        Lowest value for fracton of difference between pre-previous and previous losses
            (loss[-2] - loss[-1]) / loss[-2] < `teta`
        Can take following values:
            - None => eliminate stop criteria, all epochs will be computed
            - numeric (0.005)
    epochs : int (default 10000)
        Number of gradient steps
    batch_size : int or None (default None)

    random_state : int or None (default None)

    fit_intercept : bool (default True)
        Fit intercept or not
        Can take following values:
            - True  => y = w_0 + <X, w>
            - False => y = <X, w>
    k : (default 'inversed_epoch')
        Special coefficient in gradient step - `eta / k`
        Can take following values:
            - k='inversed_epoch' => on every step `eta` will be divided by current number of epoch
            - k=1 => fixed `eta`, setten in the beggining
    ---------------------------------------------------------------------------
    Attributes:

    losses : np.array (default [])
        Keeps history of losses after fitting
    coef : np.array (default [])
        Keeps history of coefficient sets after fitting
    '''
    def __init__(self, eta, epsilon=0.005, teta=0.005, epochs=10000,
                 batch_size=None, random_state=None,
                 fit_intercept=True, k='inversed_epoch'):
        self.eta = eta
        self.epsilon = epsilon
        self.teta = teta
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.fit_intercept = fit_intercept
        self.k = k
        self._losses = []
        self._coef = []
        
    def gradient_step(self, X, y, beta_est, k = 1):
        '''
        
        '''
        r = y - X.dot(beta_est)
        nabla = r.dot(X)
        beta_est = beta_est + self.eta / k * nabla

        self._losses.append((r ** 2).sum())
        self._coef.append(beta_est)
        
        return beta_est
        
    def fit(self, X, y):
        '''
        
        '''
        # есть ли свободный член
        if self.fit_intercept:
            # добавим колонку из единиц
            X = np.c_[np.ones(len(X)), X]
            
        # инициализация весов линейной регрессии
        beta_est = np.zeros_like(X[0])
        
        # повторить шаг градиентного спуска заданное число раз
        for epoch in range(1, self.epochs + 1):
            # критерии останова 
            if (epoch > 2):
                # если `epsilon` != None
                if self.epsilon:
                    # норма изменения коэффициентов
                    delta_coef = norm(self._coef[-1] - self._coef[-2])
                    # выход из цикла
                    if delta_coef < self.epsilon: break
                # если `teta` != None
                if self.teta:
                    # разница предпоследнего и последнего значения ошибки
                    delta_loss_frac = self._losses[-2] - self._losses[-1] / self._losses[-2]
                    # выход из цикла
                    if delta_loss_frac < self.teta: break
            # определяется, дробным шагом или фиксированным будет градиентный спуск
            # в случае 'inversed_epoch' - шаг делится на номер эпохи
            if self.k == 'inversed_epoch':
                k = epoch
            # шаг делится на 1, т.е. постоянный
            else:
                k = 1
            # mini-batch (=const), stochastic (=1)
            if self.batch_size:
                X_batch, y_batch = resample(
                    X, y, n_samples=self.batch_size,
                    replace=False, random_state=self.random_state + epoch)
            # batch (=all)
            else: 
                X_batch, y_batch = X, y
            # один шаг градиентного спуска
            beta_est = self.gradient_step(X_batch, y_batch, beta_est, k)

        # запись числа отработанных эпох в атрибут
        self.n_epoch = epoch
        # последние коэффициенты запишем в атрибут
        self.coef_fitted = beta_est
            
        # конвертация в np.array
        self._losses = np.array(self._losses)
        self._coef = np.array(self._coef)

    def predict(self, X):
        # есть ли свободный член
        if self.fit_intercept:
            # добавим колонку из единиц
            X = np.c_[np.ones(len(X)), X]
        return X.dot(self.coef_fitted)
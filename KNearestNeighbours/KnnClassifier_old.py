from KnnEstimator import KnnEstimator

import numpy as np

from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.spatial.distance import cdist
from scipy.spatial.distance import braycurtis, canberra, chebyshev, cityblock, \
                                   correlation, cosine, dice, euclidean, hamming, \
                                   jaccard, kulsinski, mahalanobis, matching, minkowski, \
                                   rogerstanimoto, russellrao, seuclidean, sokalmichener, \
                                   sokalsneath, sqeuclidean, wminkowski, yule


class KnnClassifier(KnnEstimator):
    """
    Inherited methods:
    BaseEstimator - get_params, set_params
    ClassifierMixin - score
    ------------------------
    Classifier implementing the k-nearest neighbors vote.

    Read more in the :ref:`User Guide <classification>`.

    Parameters
    ------------------------
    n_neighbors : int, optional (default = 5)
        Number of neighbors to use by default for :meth:`kneighbors` queries.

    metric : string or callable, default 'minkowski'
        the distance metric to use for the tree.  The default metric is
        minkowski, and with p=2 is equivalent to the standard Euclidean
        metric. See the documentation of the DistanceMetric class for a
        list of available metrics.

    p : integer, optional (default = 2)
        Power parameter for the Minkowski metric. When p = 1, this is
        equivalent to using manhattan_distance (l1), and euclidean_distance
        (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.

    """
    def __init__(self, n_neighbors = 5, scaler = 'StandardScaler',
                 metric = 'minkowski', p = 2, weights = 'uniform', **kwargs):

        self.n_neighbors = n_neighbors
        self.scaler = scaler
        self.metric = metric
        self.p = p
        self.weights = weights
        self.kwargs = kwargs

    def fit(self, X, y):

        self._y = y.values
        self._fit_X = X.values


    def _predict(self, X):

        # Шкалирование 3 варианта
        scalers_dict = {'StandardScaler':StandardScaler,
                        'MinMaxScaler':MinMaxScaler,
                        'None':None}

        self.scaler_function = scalers_dict[self.scaler]
        if self.scaler_function:
            scaler = self.scaler_function()
            scaler.fit(self._fit_X)
            X_train = scaler.transform(self._fit_X)
            X_test = scaler.transform(X)
        else:
            X_train = self._fit_X
            X_test = X

        metric_dict = {'braycurtis':braycurtis, 'canberra':canberra,
                       'chebyshev':chebyshev, 'cityblock':cityblock,
                       'correlation':correlation, 'cosine':cosine,
                       'dice':dice, 'euclidean':euclidean,
                       'hamming':hamming, 'jaccard':jaccard,
                       'kulsinski':kulsinski, 'mahalanobis':mahalanobis,
                       'matching':matching, 'minkowski':minkowski,
                       'rogerstanimoto':rogerstanimoto, 'russellrao':russellrao,
                       'seuclidean':seuclidean, 'sokalmichener':sokalmichener,
                       'sokalsneath':sokalsneath, 'sqeuclidean':sqeuclidean,
                       'wminkowski':wminkowski, 'yule':yule}

        metric = metric_dict[self.metric]
        distances = cdist(X_test, X_train, metric, self.p, self.weights, **self.kwargs)

        neighbors = np.argsort(distances)[:, :self.n_neighbors]
        probs_1_cl = np.array([np.mean(self._y[neighbor]) for neighbor in neighbors])
        return probs_1_cl

    def predict(self, X):
        """Predict the class labels for the provided data
        Parameters
        ----------
        X : array-like, shape (n_query, n_features), \
                or (n_query, n_indexed) if metric == 'precomputed'
            Test samples.
        Returns
        -------
        y : array of shape [n_samples] or [n_samples, n_outputs]
            Class labels for each data sample.
        """
        classes = self._predict(X).round()
        return classes

    def predict_proba(self, X):
        """
        Return probability estimates for the test data X.
        Parameters
        ----------
        X : array-like, shape (n_query, n_features), \
                or (n_query, n_indexed) if metric == 'precomputed'
            Test samples.
        Returns
        -------
        p : array of shape = [n_samples, n_classes], or a list of n_outputs
            of such arrays if n_outputs > 1.
            The class probabilities of the input samples. Classes are ordered
            by lexicographic order.
        """
        probs_1_cl = self._predict(X)
        probs_2_cl = np.array(list(zip(1 - probs_1_cl, probs_1_cl)))
        return probs_2_cl

    def score(self, X, y, sample_weight=None):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples.
        y : array-like, shape = (n_samples) or (n_samples, n_outputs)
            True labels for X.
        sample_weight : array-like, shape = [n_samples], optional
            Sample weights.
        Returns
        -------
        score : float
            f1_score of self.predict(X) wrt. y.
        """
        return f1_score(y, self.predict(X), sample_weight=sample_weight)

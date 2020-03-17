# TODO: parzen_window as a func
#       documentation

from KnnEstimator import KnnEstimator
from rbf_kernels import kernel_gauss, kernel_epanechnikov, kernel_tri_cube

import numpy as np

from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.spatial.distance import cdist
from scipy.spatial.distance import braycurtis, canberra, chebyshev, \
                                   cityblock, correlation, cosine, \
                                   dice, euclidean, hamming, jaccard, \
                                   kulsinski, mahalanobis, matching, \
                                   minkowski,rogerstanimoto, russellrao, \
                                   seuclidean,sokalmichener, sokalsneath, \
                                   sqeuclidean, wminkowski, yule
# TODO: вынести в отдельный файл?
# Шкалирование 3 варианта
_scalers_dict = {'StandardScaler':StandardScaler,
                 'MinMaxScaler':MinMaxScaler}
_metric_dict = {'braycurtis':braycurtis, 'canberra':canberra,
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
_kernel_dict = {'kernel_gauss':kernel_gauss,
                'kernel_epanechnikov':kernel_epanechnikov,
                'kernel_tri_cube':kernel_tri_cube}

class KnnClassifier(KnnEstimator):
    '''
    Classifier implementing the k-nearest neighbors vote.
    ---------------------------------------------------------------------------
    Parameters
    
    n_neighbors : int, optional (default = 5)
        Number of neighbors to use by default for :meth:`kneighbors` queries.

    scaler : string (default None)
        ...
        'StandardScaler'
        'MinMaxScaler'

    kernel : string (default None)
        ...
        'kernel_gauss'
        'kernel_epanechnikov'
        'kernel_tri_cube'

    metric : string or callable (default 'minkowski')
        the distance metric to use for the tree.  The default metric is
        minkowski, and with p=2 is equivalent to the standard Euclidean
        metric. See the documentation of the DistanceMetric class for a
        list of available metrics.

    p : integer, optional (default = 2)
        Power parameter for the Minkowski metric. When p = 1, this is
        equivalent to using manhattan_distance (l1), and euclidean_distance
        (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.

    weights : string (default 'uniform')
        ...
        'uniform'
    ---------------------------------------------------------------------------
    Inherited methods:
    BaseEstimator - get_params, set_params
    ClassifierMixin - score
    '''
    def __init__(self, n_neighbors = 5, scaler = None,
                 kernel=None, h=-1, metric = 'minkowski',
                 p = 2, weights = 'uniform', **kwargs):
        self.n_neighbors = n_neighbors
        self.scaler = scaler
        self.kernel = kernel
        self.h = h
        self.metric = metric
        self.p = p
        self.weights = weights
        self.kwargs = kwargs

    def fit(self, X, y):
        self.labels = np.unique(y)
        self._y = np.asarray(y)
        self._fit_X = np.asarray(X)

    def _predict(self, X):
        global _scalers_dict, _metric_dict, _kernel_dict

        if self.scaler:
            scaler_method = _scalers_dict[self.scaler]
            scaler = scaler_method()
            scaler.fit(self._fit_X)
            X_train = scaler.transform(self._fit_X)
            X_test = scaler.transform(X)
        else:
            X_train = self._fit_X
            X_test = X

        metric = _metric_dict[self.metric]
        distances = cdist(X_test, X_train, metric,
                          self.p, self.weights, **self.kwargs)
        # sorted distance array
        sort_distances = np.sort(distances)
        # k-nearest distancies
        knn_distances = sort_distances[:, :self.n_neighbors]
        # indicies k-neareast
        neighbors = np.argsort(distances)[:, :self.n_neighbors]
        # 
        label_masks = np.array([self._y[neighbors] == label for label in self.labels])
        
        # kernel != None
        if self.kernel:
            # parzen window
            # adaptive window
            if self.h == -1:
                # use k+1 nearest neighbor as a denominator in kernel
                h = sort_distances[:, self.n_neighbors]
                # from vector to matrix for dividing
                h = h[:, np.newaxis]
            # fixed window
            else:
                h = self.h
            # similar to normalization
            knn_distances_normalized = knn_distances / h
            # eleminate influence of points which are behind h-radius
            knn_distances_normalized[knn_distances_normalized > 1] = 1
            # choose kernel
            kernel = _kernel_dict[self.kernel]
            # compute kernel
            knn_kernels = kernel(knn_distances_normalized)

            probabilities = np.array(
                [(knn_kernels * l_mask).sum(axis=1) / knn_kernels.sum(axis=1) \
                for l_mask in label_masks]).T
        else:
            probabilities = np.array(
                [(l_mask).sum(axis=1) / self.n_neighbors \
                for l_mask in label_masks]).T

        return probabilities

    def predict(self, X):
        '''
        Predict the class labels for the provided data.
        Parameters
        ----------
        X : array-like, shape (n_query, n_features), \
                or (n_query, n_indexed) if metric == 'precomputed'
            Test samples.
        Returns
        -------
        y : array of shape [n_samples] or [n_samples, n_outputs]
            Class labels for each data sample.
        '''
        label_inds = self._predict(X).argmax(axis=1)
        classes = self.labels[label_inds]
        return classes

    def predict_proba(self, X):
        '''
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
        '''
        return self._predict(X)

    def score(self, X, y, sample_weight=None):
        '''
        Return f1_score computed between `y` and `y_hat` predicted on X.
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
        '''
        return f1_score(y, self.predict(X), sample_weight=sample_weight)

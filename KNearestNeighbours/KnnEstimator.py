from sklearn.base import BaseEstimator, ClassifierMixin

class KnnEstimator(BaseEstimator, ClassifierMixin):
    """
    Inherited methods:
    BaseEstimator - get_params, set_params
    ClassifierMixin - score
    """
    _estimator_type = "classifier"

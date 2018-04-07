from sklearn.base import BaseEstimator, ClassifierMixin

class MyNNEstimator(BaseEstimator, ClassifierMixin):
    """
    Inherited methods:
    BaseEstimator - get_params, set_params
    ClassifierMixin - score
    """
    _estimator_type = "classifier"

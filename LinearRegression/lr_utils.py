import numpy as np

def linspace_vec(X_min_max, num=50, endpoint=True, retstep=False):
    X_linspase = [
        np.linspace(arr[0], arr[1], num=num, endpoint=endpoint, retstep=retstep)
        for arr in X_min_max
    ]
    return np.array(X_linspase)

def min_max_vec(X):
    return np.c_[X.min(axis=0), X.max(axis=0)]
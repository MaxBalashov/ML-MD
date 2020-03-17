import numpy as np
from numpy.linalg import norm
# TODO: write docs
'''
Sklearn kernels:
http://scikit-learn.org/stable/modules/svm.html#kernel-functions
'''
def linear_kernel(x1, x2):
    '''
    '''
    return np.dot(x1, x2)

def gaussian_kernel(x1, x2, sigma = 1):
    '''
    Gaussian kernel (aka `radial basis function kernel`)

    https://en.wikipedia.org/wiki/Radial_basis_function_kernel
    '''
    return np.exp(-(norm(x1 - x2) ** 2 / (2 * sigma)))

def polynomial_kernel(x1, x2, degree=2, gamma=1, coef0=1):
    '''
    '''
    return (gamma * np.dot(x1, x2) + coef0) ** degree

def sigmoid_kernel(x1, x2, gamma=1, coef0=1):
    '''
    '''
    return np.tanh(gamma * np.dot(x1, x2) + coef0)
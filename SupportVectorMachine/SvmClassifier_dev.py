import numpy as np
from cvxopt import matrix, solvers
from sklearn.base import BaseEstimator, ClassifierMixin
from kernels import linear_kernel, gaussian_kernel, \
                    polynomial_kernel, sigmoid_kernel


_kernel_dict = {'linear' : linear_kernel,
                'gaussian' : gaussian_kernel,
                'polynomial' : polynomial_kernel,
                'sigmoid' : sigmoid_kernel}
# Don't show optimization progress
solvers.options['show_progress'] = False

class SvmEstimator(BaseEstimator, ClassifierMixin):
    '''
    Inherited methods:
        BaseEstimator - get_params, set_params
        ClassifierMixin - score
    '''
    _estimator_type = 'classifier'


class SvmClassifier(SvmEstimator):
    # TODO: write docs in SvmClassifier
    '''
    Classifier implementing `Support Vector Machine`.
    ---------------------------------------------------------------------------
    Parameters
    kernel : string (default None) # TODO: add **kernel_args - gamma, sigma, coef0
        Kernel for `Kernel trick`
            - 'linear'
                <x1, x2>
            - 'gaussian'
                e^(||x1 - x2||^2 / (2*sigma))
            - 'polynomial'
                (g * <x1, x2> + c)^d
            - 'sigmoid' # TODO: Bugged (kernel='sigmoid', C=None)
                tanh(g * <x1, x2> + c)
    C : float (default None)
        Constant which bounds alphas from above
            0 <= alpha_i <= C
    threshold : float (default 1e-5)
        If any alpha is less than the threshold, then it is assumed equal to zero
    kernel_args : dict() (default None)
        for 'linear' : ---
        for 'gaussian' : sigma=1
        for 'polynomial' : degree=2, gamma=1, coef0=1
        for 'sigmoid': gamma=1, coef0=1
    cvxopt_args : dict() (default None)
        'show_progress' : bool (default: True)
            turns the output to the screen on or off.
        'maxiters' : int (default: 100)
            maximum number of iterations.
        'abstol' : float (default: 1e-7)
            absolute accuracy.
        'reltol' : float (default: 1e-6)
            relative accuracy.
        'feastol' : float (default: 1e-7)
            tolerance for feasibility conditions.
        'refinement' : int (default: 0 or 1)
            number of iterative refinement steps when solving KKT equations.
            (default: 0 if the problem has no second-order cone or matrix inequality constraints; 1 otherwise)
        http://cvxopt.org/userguide/coneprog.html#algorithm-parameters
    '''
    def __init__(self, kernel='linear', C=None, threshold=1e-5, **cvxopt_args): #, **kernel_args
        self.kernel = kernel
        self.C = C
        self.threshold = threshold
        # self.kernel_args = kernel_args
        self.cvxopt_args = cvxopt_args

        self.kernel_function = _kernel_dict[kernel]#(self.kernel_args)

    def fit(self, X, y):
        n_samples = X.shape[0]
        
        # Gram matrix
        self.K = np.array([[self.kernel_function(i, j) for i in X] for j in X])
        
        # Construct matrices for Optimization problem
        # Target function (which minimizes)
        Q = matrix(self.K * np.outer(y, y))
        p = matrix(-np.ones(n_samples))
        # Equalities (but here is one)
        A = matrix(y, (1, n_samples), tc='d')
        b = matrix(0.0)
        
        # Unequalities
        if self.C is None:
            G = matrix(-np.eye(n_samples))
            h = matrix(np.zeros(n_samples))
        else:
            G = matrix(np.r_[-np.eye(n_samples),
                             np.eye(n_samples)])
            h = matrix(np.r_[np.zeros(n_samples),
                             np.full(n_samples, self.C, dtype='float32')])
        
        # Set parameters to solver
        for param, value in self.cvxopt_args.items():
            solvers.options[param] = value

        # Solve QP problem
        self.qp_solution = solvers.qp(Q, p, G, h, A, b)
        
        # Lagrange multipliers (alphas or lambdas)
        alphas = np.ravel(self.qp_solution['x'])
        
        # Support vectors have non zero lagrange multipliers
        mask_sup_vec = alphas > self.threshold
        self.alphas = alphas[mask_sup_vec]
        self.X_sv = X[mask_sup_vec]
        self.y_sv = y[mask_sup_vec]

        # TODO: write a formula
        self.bias = np.sum(self.y_sv)
        self.bias -= np.sum(
            self.K[mask_sup_vec][:, mask_sup_vec] * \
            self.alphas * \
            self.y_sv)
        self.bias /= len(self.alphas)
     
    def _predict(self, x):
        '''
        '''
        # kernels value for point by Support Vectors
        K = np.array([self.kernel_function(x_i, x) for x_i in self.X_sv])
        # sign( sum(alpha_i * y_i * K(x_i, x)) + w_0)
        y_pred = np.sign(np.dot(self.alphas * self.y_sv, K) + self.bias)
        return y_pred
        
    def predict(self, X):
        '''
        '''
        return np.array([self._predict(x) for x in X])
        
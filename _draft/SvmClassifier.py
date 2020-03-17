import numpy as np
from cvxopt import matrix, solvers

def linear_kernel(x1, x2):
    return np.dot(x1, x2)

_kernel_dict = {'linear':linear_kernel}

class SvmClassifier:
    '''
    Classifier implementing `Support Vector Machine`.
    ---------------------------------------------------------------------------
    Parameters
    kernel : string (default None)
        Kernel for `Kernel trick`
    C : float (default None)
        Constant
    '''
    def __init__(self, kernel='linear', C=None, threshold=1e-5):
        self.kernel = _kernel_dict[kernel]
        self.C = C
#         if self.C is not None: self.C = float(self.C)
        self.threshold = threshold

    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Gram matrix
        self.K = np.array([[self.kernel(i, j) for i in X] for j in X])
        
        # Construct matrices for Optimization problem
        # Target function (which minimizes)
        Q = matrix(self.K * np.outer(y, y))
        p = matrix(-np.ones(n_samples))
        
        # Unequalities
        if self.C is None:
            G = matrix(-np.eye(n_samples))
            h = matrix(np.zeros(n_samples))
        else:
            G = matrix(np.r_[-np.eye(n_samples),
                             np.eye(n_samples)])
            h = matrix(np.r_[np.zeros(n_samples),
                             np.full(n_samples, self.C, dtype='float32')])
        # Equalities (but here is one)
        A = matrix(y, (1, n_samples), tc='d')
        b = matrix(0.0)
        
        # Solve QP problem
        self.qp_solution = solvers.qp(Q, p, G, h, A, b)
        
        # Lagrange multipliers (alphas or lambdas)
        alphas = np.ravel(self.qp_solution['x'])
        
        # Support vectors have non zero lagrange multipliers
        self.mask_sup_vec = alphas > self.threshold
        self.alphas = alphas[self.mask_sup_vec]
        self.X_sv = X[self.mask_sup_vec]
        self.y_sv = y[self.mask_sup_vec]
        print("%d support vectors out of %d points" % (len(self.alphas), n_samples))

        # self.weights = \
        #     np.sum(
        #         self.K[self.mask_sup_vec][:, self.mask_sup_vec] * \
        #         self.alphas * \
        #         self.y_sv, axis=1)
        # self.bias = np.mean(self.kernel(self.X_sv, self.weights) - self.y_sv)

        self.bias = np.sum(self.y_sv)
        self.bias -= np.sum(
            self.K[self.mask_sup_vec][:, self.mask_sup_vec] * \
            self.alphas * \
            self.y_sv)
        self.bias /= len(self.alphas)
        
    def _predict(self, X):
        # kernels value for point by Support Vectors
        K = self.kernel(self.X_sv, X)
        # sign( sum(alpha_i * y_i * K(x_i, x)) - w_0)
        classes = np.sign(np.dot(self.alphas * self.y_sv, K) + self.bias)
        return classes
        
    def predict(self, X):
        return np.array([self._predict(i) for i in X])
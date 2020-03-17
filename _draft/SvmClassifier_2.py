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
    def __init__(self, kernel='linear', C=None, threshold=1e-5, sigma=1, degree=2):
        self.kernel = _kernel_dict[kernel]
        self.C = C
        if self.C is not None: self.C = float(self.C)
        self.threshold = threshold
        self.sigma = sigma
        self.degree = degree


    def build_kernel(self, X):
        self.K = np.dot(X, X.T)        
        if self.kernel == 'poly':
            self.K = (1. + 1./self.sigma * self.K)**self.degree

    def fit(self, X, y):
        n_samples = X.shape[0]
        
        # Gram matrix
        # self.K = np.array([[self.kernel(i, j) for i in X] for j in X])

        self.build_kernel(X)
        P = matrix(y*y.transpose()*self.K)        
        q = matrix(-np.ones((n_samples, 1)))
        A = matrix(y.reshape(1,n_samples), tc='d')
        b = matrix(0.0)
        if self.C is None:
            G = matrix(-np.eye(n_samples))#just a>=0
            h = matrix(np.zeros((n_samples,1)))
        else:
            G = matrix(np.concatenate((-np.eye(n_samples),np.eye(n_samples))))
            h = matrix(np.concatenate((np.zeros((n_samples,1)), self.C*np.ones((n_samples,1)))))
        
        # Solve QP problem
        self.qp_solution = solvers.qp(P, q, G, h, A, b)
        
        lambdas = np.ravel(self.qp_solution['x'])               
        positive_lambdas_ind = np.where(lambdas > self.threshold)[0]        
        self.positive_lambdas = lambdas[positive_lambdas_ind]
        self.positive_lambdas_count = len(self.positive_lambdas)        
        self.sv_x = X[positive_lambdas_ind]
        self.sv_y = y[positive_lambdas_ind]

        self.b = np.sum(self.sv_y)
        for i in range(self.positive_lambdas_count):
            self.b -= np.sum(
                self.positive_lambdas * \
                self.sv_y * \
                np.reshape(
                    self.K[positive_lambdas_ind[i], positive_lambdas_ind],
                    (self.positive_lambdas_count, 1)
                )
            )
        self.b /= self.positive_lambdas_count

    def predict(self, X):
        K = self.kernel(X, self.sv_x.T)
        y = np.zeros(np.shape(X)[0])
        for i in range(np.shape(X)[0]):
            for j in range(self.positive_lambdas_count):
                y[i] += self.positive_lambdas[j] * self.sv_y[j] * K[i,j]
            y[i]+=self.b                    
        return np.sign(y)
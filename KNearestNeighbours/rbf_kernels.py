import numpy as np

# Kernels
def kernel_gauss(dist_arr, sigma = 1):
    return np.exp(-(dist_arr ** 2 / (2 * sigma)))

def _D_epanechnikov(t):
    d = 0
    if t < 1:
        d = 3 / 4 * (1 - t ** 2)
    return d
_D_epanechnikov_vec = np.vectorize(_D_epanechnikov)

def kernel_epanechnikov(dist_arr, lmbd = 1):
    return _D_epanechnikov_vec(dist_arr / lmbd)

def _D_tri_cube(t):
    d = 0
    if t < 1:
        d = (1 - t ** 3) ** 3
    return d
_D_tri_cube_vec = np.vectorize(_D_tri_cube)

def kernel_tri_cube(dist_arr, lmbd = 1):
    return _D_tri_cube_vec(dist_arr / lmbd)
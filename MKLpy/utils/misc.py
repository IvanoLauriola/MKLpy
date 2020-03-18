import numpy as np
from sklearn.metrics.pairwise import check_pairwise_arrays


def uniform_vector(n):
    return np.ones(n)/n


def onehot_vector(n,idx):
	w = np.zeros(n)
	w[idx] = 1
	return w




def identity_kernel(n):
    return np.diag(np.ones(n))

def to_diagonal(K):
    return np.diagonal([K[i,i] for i in range(K.shape[0])])

def ideal_kernel(Y, T=None):
    """performs the ideal kernel between the labels vectors *Y* and *Z*.
    The ideal kernel kernel is defines as:
    .. math:: YT^\top

    Parameters
    ----------
    Y : (n) array_like,
        the train labels vector.
    T : (l) array_like,
        the test labels vector. If it is not defined, then the kernel is calculated
        between *Y* and *Y*.

    Returns
    -------
    K : (l,n) ndarray,
        the ideal kernel matrix.
    """
    pos = Y[0]
    yy = [1 if y==pos else -1 for y in Y]
    tt = yy if T is None else [1 if t==pos else -1 for t in T]
    return np.dot(np.array([yy]).T,np.array([tt]))








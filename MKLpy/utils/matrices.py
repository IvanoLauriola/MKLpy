import numpy as np
import types


def identity_kernel(n):
    return np.diag(np.ones(n))

def to_diagonal(K):
    return np.diagonal([K[i,i] for i in range(K.shape[0])])

def ideal_kernel(y, z=None):
    """performs the ideal kernel between the labels vectors *Y* and *Z*.
    The ideal kernel kernel is defines as:
    .. math:: YZ^\top

    Parameters
    ----------
    Y : (n) array_like,
        the train labels vector.
    Z : (l) array_like,
        the test labels vector. If it is not defined, then the kernel is calculated
        between *Y* and *Y*.

    Returns
    -------
    K : (l,n) ndarray,
        the ideal kernel matrix.
    """
    z = y if type(z) == types.NoneType else z
    return np.dot(np.array([y]).T,np.array([z]))


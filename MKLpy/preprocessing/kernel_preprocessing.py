"""
.. codeauthor:: Ivano Lauriola <ivanolauriola@gmail.com>

==================
Regularize Kernels
==================

.. currentmodule:: MKLpy.regularization

This module contains function that perform a transformation over kernels and samples matrices

"""

import numpy as np
from MKLpy.metrics import trace
from MKLpy.utils.validation import check_squared, check_X



def kernel_normalization(K):
    """normalize a squared kernel matrix

    Parameters
    ----------
    K : (n,n) ndarray,
        the squared kernel matrix.

    Returns
    -------
    Kn : ndarray,
         the normalized version of *K*.

    Notes
    -----
    Given a kernel K, the normalized version is defines as:
    
    .. math:: \hat{k}(x,z) = \frac{k(x,z)}{\sqrt{k(x,x)\cdot k(z,z)}}
    """
    K = check_squared(K)
    n = K.shape[0]
    d = np.array([[K[i,i] for i in range(n)]])
    Kn = K / np.sqrt(np.dot(d.T,d))
    return Kn
    

def tracenorm(K):
    """normalize the trace of a squared kernel matrix

    Parameters
    ----------
    K : (n,n) ndarray,
        the squared kernel matrix.

    Returns
    -------
    Kt : ndarray,
         the trace-normalized version of *K*.

    Notes
    -----
    In trace-normalization, the kernel is divided by the average of the diagonal.
    """
    K = check_squared(K)
    trn = trace(K) / K.shape[0]
    return K / trn



def kernel_centering(K):
    """move a squared kernel at the center of axis

    Parameters
    ----------
    K : (n,n) ndarray,
        the squared kernel matrix.
        
    Returns
    -------
    Kc : ndarray,
         the centered version of *K*.
    """
    K = check_squared(K)
    N = K.shape[0]
    I = np.ones(K.shape)
    C = np.diag(np.ones(N)) - (1.0/N * I)
    Kc = np.dot(np.dot(C , K) , C)
    return Kc

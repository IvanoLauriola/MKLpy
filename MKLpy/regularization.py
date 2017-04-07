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
from MKLpy.utils.validation import check_squared

def normalization(X):
    """normalize a samples matrix (n,m) .. math:: \|X_i\|^2 = 1 \forall i \in [1..n]

    Parameters
    ----------
    X : (n,m) ndarray,
        where *n* is the number of samples and *m* is the number of features.

    Returns
    -------
    Xn : (n,m) ndarray,
         the normalized version of *X*.
    """
    return np.array([x/np.linalg.norm(x) for x in X])


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



def rescale(X):
    """edit a samples matrix by rescaling the features in [-1,1]
    
    Parameters
    ----------
    X : (n,m) ndarray,
        where *n* is the number of samples and *m* is the number of features.

    Returns
    -------
    Xr : (n,m) ndarray,
         the rescaled version of *X* in [-1,1].
    """
    X = rescale_01(X)
    return (X * 2) - 1


def rescale_01(X):
    """edit a samples matrix by rescaling the features in [0,1]
    
    Parameters
    ----------
    X : (n,m) ndarray,
        where *n* is the number of samples and *m* is the number of features.

    Returns
    -------
    Xr : (n,m) ndarray,
         the rescaled version of *X* in [0,1].
    """
    d = X.shape[1]
    for i in range(d):
        mi_v = min(X[:,i])
        ma_v = max(X[:,i])
        if mi_v!=ma_v:
            X[:,i] = (X[:,i] - mi_v)/(ma_v-mi_v)
    return X


def centering(X):
    """move the data at the center of axis

    Parameters
    ----------
    X : (n,m) ndarray,
        where *n* is the number of samples and *m* is the number of features.

    Returns
    -------
    Xc : (n,m) ndarray,
         the centered version of *X*.
    """
    n = X.shape[0]
    uno = np.ones((n,1))
    Xm = 1.0/n * np.dot(uno.T,X)
    return X - np.dot(uno,Xm)


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
    N = K.shape[0] * 1.0
    I = np.ones(K.shape)
    C = np.diag(np.ones(N)) - (1/N * I)
    Kc = np.dot(np.dot(C , K) , C)
    return Kc

def edit(KL,f):
    """apply a transformation function to each kernel in list

    Parameters
    ----------
    K_list : (l,n,n) ndarray or kernel_list,
             where *l* is the number of kernels in list and *(n,n)* is the shape of each kernels.
    f : callable with a single kernel *K* as input that return a transformation of *K*.

    Returns
    -------
    K_list_t : (l,n,n) ndarray,
            that is a list composed by *f(K)* for each *K* in *K_list*
    """
    return np.array([f(K) for K in KL])







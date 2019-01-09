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
    #return np.array([x/np.linalg.norm(x) for x in X])
    check_X(X)
    return (X.T / np.linalg.norm(X,axis=1) ).T



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
    #d = X.shape[1]
    #for i in range(d):
    #    mi_v = min(X[:,i])
    #    ma_v = max(X[:,i])
    #    if mi_v!=ma_v:
    #        X[:,i] = (X[:,i] - mi_v)/(ma_v-mi_v)
    #return X

    mi, ma = np.min(X,axis=0), np.max(X,axis=0)
    d = ma-mi
    np.putmask(d, d == 0, 1)
    return (X - mi) / d


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



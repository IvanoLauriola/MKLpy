"""
.. codeauthor:: Ivano Lauriola <ivanolauriola@gmail.com>

==================
Regularize Kernels
==================

.. currentmodule:: MKLpy.regularization

This module contains function that perform a transformation over kernels and samples matrices

"""

import torch
from ..metrics import trace
from ..utils.validation import check_X


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
    X = check_X(X)
    return (X.T / torch.norm(X, dim=1, p=2)).T



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
    X = check_X(X)
    mi, ma = X.min(dim=0)[0], X.max(dim=0)[0]
    d = ma-mi
    Xr = (X - mi) / d
    Xr = (Xr * 2) - 1
    Xr[Xr != Xr] = 0
    return Xr


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

    X = check_X(X)
    mi, ma = X.min(dim=0)[0], X.max(dim=0)[0]
    d = ma-mi
    Xr = (X - mi) / d
    Xr[Xr != Xr] = 0
    return Xr


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
    
    X = check_X(X)
    n = X.size()[0]
    uno = torch.ones((n,1))
    Xm = 1.0/n * uno.T @ X
    return X - uno @ Xm



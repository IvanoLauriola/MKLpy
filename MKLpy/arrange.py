"""
.. codeauthor:: Ivano Lauriola <ivanolauriola@gmail.com>

===============
Arrange Kernels
===============

.. currentmodule:: MKLpy.arrange

This module contains functions able to combine a list of kernels in a single one.

"""

import numpy as np
import types


def summation (K_list, weights = None):
    """perform the weighted summation of each kernel in K_list.

    Parametes
    ---------
    K_list : (l,n,m) ndarray, kernel_list or array_like,
             where *l* is the number of kernels in list and *(n,m)* is the shape of kernels.
    weights : (l) array_like,
              if it is not defined, then is performed the simple summation.

    Returns
    -------
    K : (n,m) ndarray,
        the summation of *l* kernels in list.

    Examples
    --------
    >>> Klist   = np.array([HPK_kernel(X,degree=i) for i in range(1,11)])
    >>> weights = range(1,11)
    >>> Ksum    = summation(Klist)
    >>> Kwsum   = summation(Klist,weights)
    """
        
    K = np.zeros(K_list[0].shape, dtype = np.double)
    l = len(K_list)
    weights = np.ones(l, dtype = np.double) if weights is None else weights
    for w,ker in zip(weights, K_list):
        K = K + w * ker
    return K


def multiplication (K_list, weights = None):
    """perform the weighted multiplication of each kernel in K_list.

    Parametes
    ---------
    K_list : (l,n,m) ndarray or kernel_list,
             where *l* is the number of kernels in list and *(n,m)* is the shape of kernels.
    weights : (l) array_like,
              if it is not defined, then is performed the simple multiplication.

    Returns
    -------
    K : (n,m) ndarray,
        the multiplication of *l* kernels in list.

    Examples
    --------
    >>> Klist   = np.array([HPK_kernel(X,degree=i) for i in range(1,11)])
    >>> weights = range(1,11)
    >>> Ksum    = multiplication(Klist)
    >>> Kwsum   = multiplication(Klist,weights)
    """
    K = np.ones(K_list[0].shape, dtype = np.float64)
    l = len(K_list)
    weights = np.ones(l, dtype = np.double) if type(weights) == types.NoneType else weights
    for w,ker in zip(weights, K_list):
        K = K * w * ker
    return K


def average (K_list, weights = None):
    """perform the mean of kernels in K_list.

    Parametes
    ---------
    K_list : (l,n,m) ndarray or kernel_list,
             where *l* is the number of kernels in list and *(n,m)* is the shape of kernels.
    weights : (l) array_like
             if it is not defined, then is performed the simple mean.

    Returns
    -------
    K : (n,m) ndarray,
        the mean of *l* kernels in list.

    Examples
    --------
    >>> Klist   = np.array([HPK_kernel(X,degree=i) for i in range(1,11)])
    >>> weights = range(1,11)
    >>> Ksum    = mean(Klist)
    >>> Kwsum   = mean(Klist,weights)
    """
    l = len(K_list)
    weights = np.ones(l, dtype = np.double) if weights is None else weights
    K = summation(K_list, weights)
    K /= np.sum(weights)
    return K




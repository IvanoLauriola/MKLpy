"""
.. codeauthor:: Ivano Lauriola <ivanolauriola@gmail.com>

===============
Arrange Kernels
===============

.. currentmodule:: MKLpy.arrange

This module contains functions able to combine a list of kernels in a single one.

"""

import torch


def summation (KL, weights = None):
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
        
    K = torch.zeros(KL[0].size(), )
    l = len(KL)
    weights = torch.ones(l) if weights is None else weights
    if l != len(weights):
        raise ValueError('The weights vector and the kernels list are not aligned')
    for w,ker in zip(weights, KL):
        K = K + w * ker
    return K


def multiplication (KL, weights = None):
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
    K = torch.zeros(KL[0].size(), )
    l = len(KL)
    weights = torch.ones(l) if weights is None else weights
    if l != len(weights):
        raise ValueError('The weights vector and the kernels list are not aligned')
    for w,ker in zip(weights, KL):
        K = K * w * ker
    return K


def average (KL, weights = None):
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
    l = len(KL)
    weights = torch.ones(l) if weights is None else weights
    K = summation(KL, weights) / torch.sum(weights)
    return K




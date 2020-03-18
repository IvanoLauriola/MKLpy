"""
.. codeauthor:: Ivano Lauriola <ivanolauriola@gmail.com>

================
Kernel functions
================

.. currentmodule:: MKLpy.metrics.pairwise

This module contains other kernel functions

"""

import numpy as np
from scipy.sparse import issparse
from sklearn.metrics.pairwise import linear_kernel

def homogeneous_polynomial_kernel(X, Z=None, degree=2):
    """performs the HPK kernel between the samples matricex *X* and *T*.
    The HPK kernel is defines as:
    .. math:: k(x,z) = \langle x,z \rangle^d

    Parameters
    ----------
    X : (n,m) array_like,
        the train samples matrix.
    Z : (l,m) array_like,
        the test samples matrix. If it is not defined, then the kernel is calculated
        between *X* and *X*.

    Returns
    -------
    K : (l,n) ndarray,
        the HPK kernel matrix.
    """

    return linear_kernel(X,Z)**degree
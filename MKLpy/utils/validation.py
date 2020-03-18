# -*- coding: latin-1 -*-
"""
.. codeauthor:: Ivano Lauriola <ivanolauriola@gmail.com>

================
Input validation
================

.. currentmodule:: MKLpy.utils.validation

This sub-package contains tools to check the input of a MKL algorithm.

"""
from scipy.sparse import issparse
from sklearn.utils import check_array
from sklearn.utils import column_or_1d, check_X_y
from sklearn.utils.multiclass import type_of_target
from cvxopt import matrix
import numpy as np
from .exceptions import SquaredKernelError, InvalidKernelsListError, BinaryProblemError
from ..generators import Lambda_generator


def check_X(X):
    '''check a examples matrix X'''
    check_array(X)


def check_squared(K):
    '''check if a kernel matrix K is squared'''
    check_array(K)
    if K.shape[0] != K.shape[1]:
        raise SquaredKernelError(K.shape)
    return K


def check_X_Z(X,Z):
    Z = X if Z is None else Z
    return X,Z


def check_K_Y(K,Y, binary=False):
    '''check if a kernel matrix K and labels vector Y are aligned'''
    K = check_squared(K)
    K,Y = check_X_y(K,Y)
    c = len(np.unique(Y))
    if binary and c != 2:
        raise BinaryProblemError(c)
    return K,Y


def check_KL(KL):
    '''check if KL is a kernels list'''
    if not hasattr(KL,'__len__'):
        raise InvalidKernelsListError()
    if isinstance(KL, Lambda_generator):
        return KL
    check_squared(KL[0])
    for K in KL:
        assert K.shape == KL[0].shape
    return KL


def check_KL_Y(KL,Y):
    '''check if a squared kernels list KL and a label vector Y are aligned'''
    KL = check_KL(KL)
    _,Y = check_X_y(KL[0],Y)
    return KL,Y




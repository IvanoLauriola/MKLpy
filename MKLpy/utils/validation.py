# -*- coding: latin-1 -*-
"""
.. codeauthor:: Ivano Lauriola <ivanolauriola@gmail.com>

================
Input validation
================

.. currentmodule:: MKLpy.utils.validation

This sub-package contains tools to check the input of MKL algorithms.

"""

from .exceptions import SquaredKernelError, InvalidKernelsListError, BinaryProblemError
import torch


def check_X(X):
    if type(X) != torch.Tensor:
        X = torch.tensor(X)
    if len(X.size()) != 2:
        raise ValueError('Wrong shape', X.size())
    X = X.type(torch.float64)
    return X


def check_pairwise_X_Z(X,Z):
    '''checks if X and Z are two broadcastable matrices'''
    Z = X if (Z is None) or (Z is X) else Z
    X = check_X(X)
    Z = check_X(Z)
    if X.size()[1] != Z.size()[1]:
        raise ValueError('X and Z have different features')
    return X, Z


def check_K(K):
    '''checks if a kernel matrix K is squared'''
    K = check_X(K)
    if K.size()[0] != K.size()[1] :
        raise SquaredKernelError(K.size())
    return K


def check_K_Y(K,Y, binary=False):
    '''check if a kernel matrix K and labels vector Y are aligned'''
    K = check_K(K)
    if K.size()[0] != len(Y):
        raise ValueError('The kernel matrix and the labels vector are not aligned')
    if type(Y) != torch.Tensor:
        Y = torch.tensor(Y)
    c = len(Y.unique())
    if binary and c != 2:
        raise BinaryProblemError(c)
    return K, Y


def check_KL(KL):
    '''check if KL is a kernels list'''
    if not hasattr(KL,'__len__'):
        raise InvalidKernelsListError()
    return KL


def check_KL_Y(KL, Y):
    '''check if KL is a kernels list'''
    KL = check_KL(KL)
    if 'Generator' not in type(KL).__name__:
        _, Y = check_K_Y(KL[0], Y)
    return KL, Y
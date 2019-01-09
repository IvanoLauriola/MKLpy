# -*- coding: latin-1 -*-
"""
.. codeauthor:: Ivano Lauriola <ivanolauriola@gmail.com>

================
Input validation
================

.. currentmodule:: MKLpy.utils.validation

This sub-package contains tool to check the input of a MKL algorithm.

"""
from scipy.sparse import issparse
from sklearn.utils import check_array
from sklearn.utils import column_or_1d, check_X_y
from cvxopt import matrix
import numpy as np
import types
from .exceptions import SquaredKernelError
 


def check_is_matrix(M):
    '''check if M is a 2d matrix'''
    check_array(M)

def check_X(X):
    '''check a examples matrix X'''
    check_array(X)


def check_squared(K):
    '''check if a kernel matrix K is squared'''
    check_is_matrix(K)
    if K.shape[0] != K.shape[1]:
        raise SquaredKernelError(K.shape)
    return K.todense() if issparse(K) else K

def check_X_T(X,T):
    #T = X if type(T) == types.NoneType else T
    T = T if T is not None else X
    return X,T

def check_K_Y(K,Y,binary=False):
    '''check if a kernel matrix K and labels vector Y are aligned'''
    K = check_squared(K)
    K,Y = check_X_y(K,Y)
    if binary:
        n_classes = len(np.unique(Y))
        if n_classes > 2:
            raise BinaryProblemError(n_classes)
    return K,Y


def check_KL(KL):
    '''check if KL is a kernels list'''
    if not hasattr(KL,'__len__'):
        raise TypeError("list of kernels must be array-like")
    a = KL[0]
    #check_X_y(a.T,Y, accept_sparse='csr', dtype=np.float64, order="C")
    if len(a.shape) == 1:
        raise TypeError("Expected a list of kernels, found matrix")
    if len(a.shape) != 2:
        raise TypeError("Expected a list of kernels, found unknown")
    
    #if a.shape != (KL.shape[1],KL.shape[2]):
    #    raise TypeError("Incompatible dimensions")
    return KL

def check_KL_Y(KL,Y):
    '''check if a squared kernels list KL and a label vector Y are aligned'''
    KL = check_KL(KL)
    #if KL.shape[2] != len(Y):   #TODO: use sklearn functions
    #    raise TypeError('KL and Y are not aligned')
    return KL,Y




def process_list(X,gen=None, T=None):
    '''if X is a samples matrix, then generate a kernels list according to gen object'''
    x0 = np.array(X[0]) # X può essere pure una lista
    if len(x0.shape) == 2 :
        KL = check_KL(X)  #KL can be a List and not ndarray
    else :
        T = X if type(T) == types.NoneType else T
        gen = gen if gen else HPK_generator(n=10)
        KL = gen.make_a_list(X,T)   # TOTO: use a generator to make a list
    return KL


"""
.. codeauthor:: Ivano Lauriola <ivanolauriola@gmail.com>
.. codeauthor:: Mirko Polato

================
Kernel functions
================

.. currentmodule:: MKLpy.metrics.pairwise

This module contains all kernel functions of MKLpy. These kernels can be classified in
tho classes:
* kernels used in learning phases, such as HPK as SSK;
* kernels used in evaluation, such as identity and ideal kernel.

"""

import numpy as np
import cvxopt as co #TODO: remove this, using only numpy
from scipy.special import binom
from scipy.sparse import issparse
from MKLpy.utils import bignom
from MKLpy.utils.validation import check_X_T

def HPK_kernel(X, T=None, degree=2):
    """performs the HPK kernel between the samples matricex *X* and *T*.
    The HPK kernel is defines as:
    .. math:: k(x,z) = \langle x,z \rangle^d

    Parameters
    ----------
    X : (n,m) array_like,
        the train samples matrix.
    T : (l,m) array_like,
        the test samples matrix. If it is not defined, then the kernel is calculated
        between *X* and *X*.

    Returns
    -------
    K : (l,n) ndarray,
        the HPK kernel matrix.
    """
    
    X, T = check_X_T(X, T)
    return np.dot(X,T.T)**degree


def SSK_kernel(X, T=None, k=2):
    """performs the SSK kernel between the samples matricex *X* and *T*.
    Note that this is a kernel for STRINGS and text categorization.

    Parameters
    ----------
    X : (n,m) array_like,
        the train samples matrix.
    T : (l,m) array_like,
        the test samples matrix. If it is not defined, then the kernel is calculated
        between *X* and *X*.

    Returns
    -------
    K : (l,n) ndarray,
        the SSK kernel matrix.
    """
    #TODO
    return





#----------BOOLEAN KERNELS----------

def monotone_conjunctive_kernel(X,T=None,c=2):
    T = X if type(T) == types.NoneType else T
    L = np.dot(X,T.T)
    return binom(L,c)

def monotone_disjunctive_kernel(X,T=None,d=2):
    T = X if type(T) == types.NoneType else T
    L = np.dot(X,T.T)
    n = X.shape[1]

    XX = np.dot(X.sum(axis=1).reshape(X.shape[0],1), np.ones((1,T.shape[0])))
    TT = np.dot(T.sum(axis=1).reshape(T.shape[0],1), np.ones((1,X.shape[0])))
    N_x = n - XX
    N_t = n - TT
    N_xz = N_x - TT.T + L

    N_d = binom(n, d)
    N_x = binom(N_x,d)
    N_t = binom(N_t,d)
    N_xz = binom(N_xz,d)
    return (N_d - N_x - N_t.T + N_xz)


def monotone_dnf_kernel(X,T=None,d=2,c=2):
    T = X if type(T) == types.NoneType else T
    n = X.shape[1]
    n_c = binom(n,c)
    XX = np.dot(X.sum(axis=1).reshape(X.shape[0],1), np.ones((1,T.shape[0])))
    TT = np.dot(T.sum(axis=1).reshape(T.shape[0],1), np.ones((1,X.shape[0])))
    XXc = binom(XX,c)
    TTc = binom(TT,c)
    return binom(n_c,d) - binom(n_c - XXc, d) - binom(n_c - TTc.T, d) + binom(my_mdk(X,T,c),d)

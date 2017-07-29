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
    
    if degree < 1:
        raise ValueError('degree must be greather than 0')
    if degree != floor(degree):
        raise ValueError('degree must be int')
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

#TODO: adding support to train/test matrices



def disjunctive_kernel(X, k=2):
    X, _ = check_X_T(X, None)
    X = (X > 0) * 1.0
    #print X.shape
    m, n = X.shape
    x_choose_k = [0] * (n + 1)
    for i in range(1, n + 1):
        x_choose_k[i] = binom(i, k)
    N = (2 ** k - 2) * x_choose_k[n]

    L = np.dot(X,X.T)
    K = np.zeros(L.shape)
    UU = np.ones(X.shape)-X
    for i in range(m):
        for j in range(i, m):
            r = int(   np.dot(UU[i], (UU[j]).T)[0]   )
            K[i, j] = K[j, i] = N + x_choose_k[int(L[i, j]) + r]

    return K


def monotone_disjunctive_kernel(X, k=2):
    X = X.toarray() if issparse(X) else X
    #TODO: train/test
    X, _ = check_X_T(X, None)
	#TODO: check warnings about binarization
    R = np.array((X.T > 0) * 1.0)
    n,m = R.shape

    x_choose_k = [0] * (n + 1)
    x_choose_k[0] = 0
    for i in range(1, n + 1):
        x_choose_k[i] = binom(i, k)

    nCk = x_choose_k[n]
    X = np.dot(R.T,R)
    K = np.zeros(X.shape)
    for i in range(m):
        for j in range(i, m):
            n_niCk = x_choose_k[n - int(X[i, i])]
            n_njCk = x_choose_k[n - int(X[j, j])]
            n_ni_nj_nijCk = x_choose_k[n - int(X[i, i]) - int(X[j, j]) + int(X[i, j])]
            K[i, j] = K[j, i] = nCk - n_niCk - n_njCk + n_ni_nj_nijCk
    return K


def conjunctive_kernel(M, k=2):
    M = M.toarray() if issparse(M) else M
    M = ( M > 0) * 1.0
    R = co.matrix(M.T)
    n, m = R.size
    X = R.T * R
    U = co.matrix(1.0, (n, 1))
    x_choose_k = [0] * (n + 1)
    for i in range(1, n + 1):
        x_choose_k[i] = binom(i, k)

    K = co.matrix(0.0, (m, m))
    for i in range(m):
        for j in range(i, m):
            r = int(((U - R[:, i]).T * ((U - R[:, j])))[0])
            K[i, j] = K[j, i] = x_choose_k[int(X[i, j]) + r]

    return np.array(K)


def monotone_conjunctive_kernel(X, k=2):
    X = X.toarray() if issparse(X) else X
    #TODO: train/test
    #X, _ = check_X_T(X, None)
	#TODO: check warnings about binarization
    X = np.array((X > 0) * 1.0)

    m, n = X.shape
    L = np.dot(X, X.T)

    x_choose_k = [0] * (n + 1)
    for i in range(1, n + 1):
        x_choose_k[i] = binom(i, k)

    K = np.zeros(L.shape)
    for i in range(m):
        for j in range(i, m):
            K[i, j] = K[j, i] = x_choose_k[int(L[i, j])]

    return K


def dnf_kernel(M, k, d):
    M = M.toarray() if issparse(M) else M
    M = ( M > 0) * 1.0
    R = co.matrix(M.T)
    n, m = R.size
    X = R.T * R

    C = binom(2 ** d * int(binom(n, d)), k) - 2 * binom((2 ** d - 1) * int(binom(n, d)), k)
    N = (2 ** d - 2) * int(binom(n, d))

    U = co.matrix(1.0, (n, 1))
    K = co.matrix(0.0, (m, m))
    for i in range(m):
        for j in range(i, m):
            nij = int(X[i, j]) + ((U - R[:, i]).T * ((U - R[:, j])))[0]
            K[i, j] = K[j, i] = C + binom(N + binom(nij, d), k)

    return np.array(K)


def monotone_dnf_kernel(M, k, s):
    M = M.toarray() if issparse(M) else M
    M = ( M > 0) * 1.0
    R = co.matrix(M.T)
    n, m = R.size

    x_choose_s = {n: bignom(n, s)}
    nCs = x_choose_s[n]

    x_choose_k = {nCs: bignom(nCs, k)}
    a = x_choose_k[nCs]

    X = R.T * R

    if k == s == 1:
        K = X
    else:
        K = co.matrix(0.0, (m, m))
        for i in range(m):

            for j in range(i, m):

                xii = int(X[i, i])
                if xii not in x_choose_s:
                    x_choose_s[xii] = bignom(xii, s)
                nCs_niCs = nCs - x_choose_s[xii]

                if nCs_niCs not in x_choose_k:
                    x_choose_k[nCs_niCs] = bignom(nCs_niCs, k)
                b = x_choose_k[nCs_niCs]

                xjj = int(X[j, j])
                if xjj not in x_choose_s:
                    x_choose_s[xjj] = bignom(xjj, s)
                nCs_njCs = nCs - x_choose_s[xjj]

                if nCs_njCs not in x_choose_k:
                    x_choose_k[nCs_njCs] = bignom(nCs_njCs, k)
                c = x_choose_k[nCs_njCs]

                xij = int(X[i, j])
                if xij not in x_choose_s:
                    x_choose_s[xij] = bignom(xij, s)
                nCs_niCs_njCs_nijCs = nCs - x_choose_s[xii] - x_choose_s[xjj] + x_choose_s[xij]

                if nCs_niCs_njCs_nijCs not in x_choose_k:
                    x_choose_k[nCs_niCs_njCs_nijCs] = bignom(nCs_niCs_njCs_nijCs, k)
                d = x_choose_k[nCs_niCs_njCs_nijCs]

                K[i, j] = K[j, i] = float(a - c + d - b)

    return np.array(K)


def cnf_kernel(M, d, c):
    M = M.toarray() if issparse(M) else M
    M = ( M > 0) * 1.0
    R = co.matrix(M.T)
    n, m = R.size

    x_choose_k = [0] * (n + 1)
    for i in range(1, n + 1):
        x_choose_k[i] = binom(i, d)

    N = (2 ** d - 2) * x_choose_k[n]

    X = R.T * R
    U = co.matrix(1.0, (n, 1))
    K = co.matrix(0.0, (X.size[0], X.size[1]))
    for i in range(m):
        for j in range(i, m):
            r = int(((U - R[:, i]).T * ((U - R[:, j])))[0])
            K[i, j] = K[j, i] = binom(N + x_choose_k[int(X[i, j]) + r], c)

    return np.array(K)


def monotone_cnf_kernel(M, d, c):
    M = M.toarray() if issparse(M) else M
    M = ( M > 0) * 1.0
    R = co.matrix(M.T)
    n, m = R.size

    x_choose_d = {n: binom(n, d)}
    nCd = x_choose_d[n]

    X = R.T * R

    if c == d == 1:
        K = X
    else:
        K = co.matrix(0.0, (m, m))
        for i in range(m):

            for j in range(i, m):

                xii = n - int(X[i, i])
                if xii not in x_choose_d:
                    x_choose_d[xii] = binom(xii, d)

                xjj = n - int(X[j, j])
                if xjj not in x_choose_d:
                    x_choose_d[xjj] = binom(xjj, d)

                xij = n - int(X[i, i]) - int(X[j, j]) + int(X[i, j])
                if xij not in x_choose_d:
                    x_choose_d[xij] = binom(xij, d)

                r = nCd - x_choose_d[xii] - x_choose_d[xjj] + x_choose_d[xij]
                K[i, j] = K[j, i] = binom(r, c)

    return np.array(K)
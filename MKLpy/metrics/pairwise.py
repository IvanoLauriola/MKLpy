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
from math import floor
from sklearn.metrics.pairwise import cosine_similarity
import cvxopt as co
from scipy.special import binom
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
    
    if degree < 1:
        raise ValueError('degree must be greather than 0')
    if degree != floor(degree):
        raise ValueError('degree must be int')
    X,T = check_X_T(X,T)
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



def d_kernel(X, k=2):
	X,_ = check_X_T(X,None)
	R = co.matrix((X.T>0) * 1.0)
	n = R.size[0]
	m = R.size[1]
	
	x_choose_k = [0]*(n+1)
	x_choose_k[0] = 0
	for i in range(1, n+1):
		x_choose_k[i] = binom(i,k)
	
	nCk = x_choose_k[n]
	X = R.T*R
	
	K = co.matrix(0.0, (X.size[0], X.size[1]))
	for i in range(m):
		for j in range(i, m):
			n_niCk = x_choose_k[n-int(X[i,i])]
			n_njCk = x_choose_k[n-int(X[j,j])]
			n_ni_nj_nijCk = x_choose_k[n-int(X[i,i])-int(X[j,j])+int(X[i,j])]
			K[i,j] = K[j,i] = nCk - n_niCk - n_njCk + n_ni_nj_nijCk
	return np.array(K)


'''from sklearn.datasets import load_iris
data = load_iris()
X = data.data
X -= np.mean(X)
K = d_kernel(X)
print K.shape
print K
print np.max(K)
print np.min(K)
'''


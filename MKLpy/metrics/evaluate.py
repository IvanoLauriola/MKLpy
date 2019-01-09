"""
.. codeauthor:: Ivano Lauriola <ivanolauriola@gmail.com>

==========
Evaluation
==========

.. currentmodule:: MKLpy.metrics.evaluate

This module contains functions that given a kernel returns a value calculated
with a metric.
These functions can be used also in heuristic algorithms, for example we can
assign a value to each kernel in list using the radius of MEB, or the margin.

"""

import numpy as np
from cvxopt import matrix,solvers,spdiag
from MKLpy.utils.validation import check_squared, check_K_Y

def radius(K):
    """evaluate the radius of the MEB (Minimum Enclosing Ball) of examples in
    feature space.

    Parameters
    ----------
    K : (n,n) ndarray,
        the kernel that represents the data.

    Returns
    -------
    r : np.float64,
        the radius of the minimum enclosing ball of examples in feature space.
    """
    check_squared(K)
    n = K.shape[0]
    P = 2 * matrix(K)
    p = -matrix([K[i,i] for i in range(n)])
    G = -spdiag([1.0] * n)
    h = matrix([0.0] * n)
    A = matrix([1.0] * n).T
    b = matrix([1.0])
    solvers.options['show_progress']=False
    sol = solvers.qp(P,p,G,h,A,b)
    return np.sqrt(abs(sol['primal objective']))

def margin(K,Y):
    """evaluate the margin in a classification problem of examples in feature space.
    If the classes are not linearly separable in feature space, then the
    margin obtained is 0.

    Note that it works only for binary tasks.

    Parameters
    ----------
    K : (n,n) ndarray,
        the kernel that represents the data.
    Y : (n) array_like,
        the labels vector.
    """
    K, Y = check_K_Y(K,Y,binary=True)
    n = Y.shape[0]
    Y = [1 if y==Y[0] else -1 for y in Y]
    YY = spdiag(Y)
    P = 2*(YY*matrix(K)*YY)
    p = matrix([0.0]*n)
    G = -spdiag([1.0]*n)
    h = matrix([0.0]*n)
    A = matrix([[1.0 if Y[i]==+1 else 0 for i in range(n)],
                [1.0 if Y[j]==-1 else 0 for j in range(n)]]).T
    b = matrix([[1.0],[1.0]],(2,1))
    solvers.options['show_progress']=False
    sol = solvers.qp(P,p,G,h,A,b)
    return np.sqrt(sol['primal objective'])

def ratio(K,Y):
    """evaluate the ratio between the radius of MEB and the margin in feature space.
    this ratio is defined as
    .. math:: \frac{R^2}{n\cdot\rho^2}

    Parameters
    ----------
    K : (n,n) ndarray,
        the kernel that represents the data.
    Y : (n) array_like,
        the labels vector.

    Returns
    -------
    v : np.float64,
        the value of the ratio
    """
    K, Y = check_K_Y(K,Y)
    n = len(Y)
    r2 = radius(K)**2
    m2 = (margin(K,Y)*1)**2
    return (r2/m2)/n
    return ((radius(K)**2)/(margin(K,Y)**2))/n


def trace(K):
    """return the trace of the kernel as input.

    Parameters
    ----------
    K : (n,n) ndarray,
        the kernel that represents the data.

    Returns
    -------
    t : np.float64,
        the trace of *K*
    """
    check_squared(K)
    return sum([K[i,i] for i in range(K.shape[0])])

def frobenius(K):
    """return the frobenius-norm of the kernel as input.

    Parameters
    ----------
    K : (n,n) ndarray,
        the kernel that represents the data.

    Returns
    -------
    t : np.float64,
        the frobenius-norm of *K*
    """
    check_squared(K)
    return np.sqrt(np.sum(K**2))

def spectral_ratio(K,norm=True):
    """return the spectral ratio of the kernel as input.

    Parameters
    ----------
    K : (n,n) ndarray,
        the kernel that represents the data.
    norm : bool=True,
           True if we want the normalized spectral ratio.
    
    Returns
    -------
    t : np.float64,
        the spectral ratio of *K*, normalized iif *norm=True*
    """
    check_squared(K)
    n = K.shape[0]
    c = trace(K)/frobenius(K)
    return (c-1)/(np.sqrt(n)-1) if norm else c





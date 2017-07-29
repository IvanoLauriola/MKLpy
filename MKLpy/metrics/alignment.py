"""
.. codeauthor:: Ivano Lauriola <ivanolauriola@gmail.com>

=========
Alignment
=========

.. currentmodule:: MKLpy.metrics.alignment

This module contains functions that given two or mode kernels returns
a value calculated by a metric, such as kernel alignment.

"""

import numpy as np
from MKLpy.utils.validation import check_squared, check_K_Y
from MKLpy.utils.matrices import ideal_kernel,identity_kernel

def alignment (K1, K2):
    """evaluate the kernel alignment between two kernels.

    Parameters
    ----------
    K1 : (n,n) ndarray,
          the first kernel used to evaluate the alignment.
    K2 : (n,n) ndarray,
         the last kernel usedto evaluate the alignment.

    Returns
    -------
    v : np.float64,
        the value of kernel alignment between *K1* and *K2*.
    """
    #K1 = check_squared(K1)
    #K2 = check_squared(K2)
    #def ff(k1,k2):
    #    n = len(k1)
    #    s = 0
    #    for i in xrange(n):
    #        for j in xrange(n):
    #            s += k1[i,j]*k2[i,j]
    #    return s
    f0 = np.sum(K1*K2)#ff(K1,K2)
    f1 = np.sum(K1*K1)#ff(K1,K1)
    f2 = np.sum(K2*K2)#ff(K2,K2)
    return (f0 / np.sqrt(f1*f2))


def alignment_ID(K):
    """evaluate the kernel alignment between a kernel as input and an identity kernel.

    Parameters
    ----------
    K : (n,n) ndarray,
        the kernel to evauluate the alignment.

    Returns
    -------
    v : np.float64,
        the value of kernel alignment between *K* and an identity kernel.
    """
    return alignment(K,np.diag(np.ones(K.shape[0])))

def alignment_yy(K,y1,y2=None):
    """evaluate the kernel alignment between a kernel as input and
        the ideal kernel, calculated as
        .. math:: Y\cdot Y^\top

    Parameters
    ----------
    K : (n,n) ndarray,
        the kernel to evaluate the alignment.
    Y : the labels vector, used to calculate the ideal kernel.

    Returns
    -------
    v : np.float64,
        the value of kernel alignment between *K* and YY'
    """
    return alignment(K,ideal_kernel(y1,y2))

#def centered_alignment(K1,K2):
#    C1 = kernel_centering(K1.copy())
#    C2 = kernel_centering(K2.copy())
#    return alignment(C1,C2)

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
from ..utils import validation # import check_squared, check_K_Y
from ..utils.misc import ideal_kernel,identity_kernel

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
    K1 = validation.check_squared(K1)
    K2 = validation.check_squared(K2)
    f0 = (K1*K2).sum()
    f1 = (K1*K1).sum()
    f2 = (K2*K2).sum()
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
    K = validation.check_squared(K)
    return alignment(K, identity_kernel(K.shape[0]))

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
    #print (ideal_kernel(y1), y1)
    return alignment(K,ideal_kernel(y1,y2))

#def centered_alignment(K1,K2):
#    C1 = kernel_centering(K1.copy())
#    C2 = kernel_centering(K2.copy())
#    return alignment(C1,C2)

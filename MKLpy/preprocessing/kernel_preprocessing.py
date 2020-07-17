# -*- coding: latin-1 -*-

"""
@author: Ivano Lauriola
@email: ivano.lauriola@phd.unipd.it, ivanolauriola@gmail.com

This file is part of MKLpy: a scikit-compliant framework for Multiple Kernel Learning
This file is distributed with the GNU General Public License v3 <http://www.gnu.org/licenses/>.  

"""

import numpy as np
from ..metrics import trace
from ..utils.validation import check_K




def kernel_normalization(K):
    """normalize a squared kernel matrix

    Parameters
    ----------
    K : (n,n) ndarray,
        the squared kernel matrix.

    Returns
    -------
    Kn : ndarray,
         the normalized version of *K*.

    Notes
    -----
    Given a kernel K, the normalized version is defines as:
    
    .. math:: \hat{k}(x,z) = \frac{k(x,z)}{\sqrt{k(x,x)\cdot k(z,z)}}
    """

    K   = check_K(K)
    n = K.size()[0]
    d = K.diag().view(n,1)
    K /= (d @ d.T)**0.5
    K[K!=K] = 0
    return K
    





def tracenorm(K):
    """normalize the trace of a squared kernel matrix

    Parameters
    ----------
    K : (n,n) ndarray,
        the squared kernel matrix.

    Returns
    -------
    Kt : ndarray,
         the trace-normalized version of *K*.

    Notes
    -----
    In trace-normalization, the kernel is divided by the average of the diagonal.
    """
    K = check_K(K)
    trn = trace(K) / K.size()[0]
    return K / trn



def kernel_centering(K):
    """move a squared kernel at the center of axis

    Parameters
    ----------
    K : (n,n) ndarray,
        the squared kernel matrix.
        
    Returns
    -------
    Kc : ndarray,
         the centered version of *K*.
    """
    K = check_K(K)
    N = K.size()[0]
    I = torch.ones(K.size())
    C = torch.ones(N).diag() - (1.0/N * I)
    return C @ K @ C

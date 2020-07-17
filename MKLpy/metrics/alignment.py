# -*- coding: latin-1 -*-

"""
@author: Ivano Lauriola
@email: ivano.lauriola@phd.unipd.it, ivanolauriola@gmail.com

This file is part of MKLpy: a scikit-compliant framework for Multiple Kernel Learning
This file is distributed with the GNU General Public License v3 <http://www.gnu.org/licenses/>.  

"""

from ..utils import validation
from ..utils.misc import ideal_kernel, identity_kernel

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

    K1 = validation.check_K(K1)
    K2 = validation.check_K(K2)
    f0 = (K1*K2).sum()
    f1 = (K1*K1).sum()
    f2 = (K2*K2).sum()
    return (f0 / (f1*f2)**.5).item()


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

    K = validation.check_K(K)
    return alignment(K, identity_kernel(K.shape[0]))

def alignment_yy(K,y1,y2=None):
    """evaluate the kernel alignment between a kernel as input and
        the ideal kernel, calculated as
        .. math:: YY^\top

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

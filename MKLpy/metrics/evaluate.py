# -*- coding: latin-1 -*-

"""
@author: Ivano Lauriola
@email: ivano.lauriola@phd.unipd.it, ivanolauriola@gmail.com

This file is part of MKLpy: a scikit-compliant framework for Multiple Kernel Learning
This file is distributed with the GNU General Public License v3 <http://www.gnu.org/licenses/>.  

"""

from cvxopt import matrix,solvers,spdiag
from ..utils import validation
from sklearn.svm import SVC
import torch
import numpy as np

_solvers = ['cvxopt', 'libsvm']


def radius(K, Y=None):
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

    K = validation.check_K(K).numpy()
    n = K.shape[0]
    P = 2 * matrix(K)
    p = -matrix(K.diagonal())
    G = -spdiag([1.0] * n)
    h = matrix([0.0] * n)
    A = matrix([1.0] * n).T
    b = matrix([1.0])
    solvers.options['show_progress']=False
    sol = solvers.qp(P,p,G,h,A,b)
    return abs(sol['primal objective'])**.5




def margin(K,Y, return_coefs=False, init_vals=None, solver='cvxopt', max_iter=-1, tol=1e-6):

    K, Y = validation.check_K_Y(K, Y, binary=True)
    Y = torch.tensor([1 if y==Y[0] else -1 for y in Y])
    params = {'K':K, 'Y':Y, 'init_vals':init_vals, 'max_iter':max_iter, 'tol':tol}

    if solver == 'cvxopt':
    	obj, gamma = _margin_cvxopt(**params)
    elif solver == 'libsvm':
    	obj, gamma = _margin_libsvm(**params)
    else:
    	raise ValueError('solver not found. Available solvers are:', _solvers)
    return (obj, gamma) if  return_coefs else obj


def _margin_cvxopt(K, Y, init_vals=None, max_iter=-1, tol=1e-6):
    '''margin optimization with CVXOPT'''

    n = Y.size()[0]
    YY = spdiag(Y.numpy().tolist())
    P = 2*(YY*matrix(K.numpy())*YY)
    p = matrix([0.0]*n)
    G = -spdiag([1.0]*n)
    h = matrix([0.0]*n)
    A = matrix([[1.0 if Y[i]==+1 else 0 for i in range(n)],
                [1.0 if Y[j]==-1 else 0 for j in range(n)]]).T
    b = matrix([[1.0],[1.0]],(2,1))
    solvers.options['show_progress']=False
    if max_iter > 0:
    	solvers.options['maxiters']=max_iter
    solvers.options['abstol']=tol
    sol = solvers.qp(P,p,G,h,A,b, initvals=init_vals)
    gamma = torch.Tensor(np.array(sol['x'])).double().T[0]
    return sol['primal objective']**.5, gamma


def _margin_libsvm(K, Y, init_vals=None, max_iter=-1, tol=1e-6):
    '''margin optimization with libsvm'''

    svm = SVC(C=1e7, kernel='precomputed', tol=tol, max_iter=max_iter).fit(K,Y)
    n = len(Y)
    gamma = torch.zeros(n).double()
    gamma[svm.support_] = torch.tensor(svm.dual_coef_)
    idx_pos = gamma > 0
    idx_neg = gamma < 0
    sum_pos, sum_neg = gamma[idx_pos].sum(), gamma[idx_neg].sum()
    gamma[idx_pos] /= sum_pos
    gamma[idx_neg] /= sum_neg
    gammay = gamma * Y
    obj = (gammay.view(n,1).T @ K @ gammay).item() **.5
    return obj, gamma




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
    K, Y = validation.check_K_Y(K, Y, binary=True)
    r2 = radius(K)**2
    m2 = margin(K,Y)**2
    return (r2/m2)/len(Y)
    #return ((radius(K)**2)/(margin(K,Y)**2))/n


def trace(K, Y=None):
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
    K = validation.check_K(K)
    return K.diag().sum().item()

def frobenius(K, Y=None):
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
    K = validation.check_K(K)
    return ( (K**2).sum()**.5 ).item()

def spectral_ratio(K, Y=None, norm=True):
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
    K = validation.check_K(K)
    n = K.size()[0]
    c = trace(K)/frobenius(K)
    return (c-1)/(n**.5 - 1) if norm else c





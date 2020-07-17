# -*- coding: latin-1 -*-

"""
@author: Ivano Lauriola
@email: ivano.lauriola@phd.unipd.it, ivanolauriola@gmail.com

This file is part of MKLpy: a scikit-compliant framework for Multiple Kernel Learning
This file is distributed with the GNU General Public License v3 <http://www.gnu.org/licenses/>.  

"""

import torch
from ...utils.validation import check_pairwise_X_Z
#from ...ultils import validation as v# import check_pairwise_X_Z





def linear_kernel(X, Z=None, normalize=False):
    """computes the linear kernel between the samples matrices *X* and *Z*.
    The kernel is defined as k(x,z) = <x,z>
    
    Parameters
    ----------
    X: torch tensor of shape (n_samples_1, n_features)
    Z: torch tensor of shape (n_samples_2, n_features)

    Returns
    -------
    K: torch tensor of shape (n_samples_1, n_samples_2),
        the kernel matrix.
    """

    X, Z = check_pairwise_X_Z(X, Z)
    return X @ Z.T


def homogeneous_polynomial_kernel(X, Z=None, degree=2):
    """computes the homogeneous polynomial kernel (HPK) between the samples matrices *X* and *Z*.
    The kernel is defined as k(x,z) = <x,z>**degree
    
    Parameters
    ----------
    X: torch tensor of shape (n_samples_1, n_features)
    Z: torch tensor of shape (n_samples_2, n_features)

    Returns
    -------
    K: torch tensor of shape (n_samples_1, n_samples_2),
        the kernel matrix.
    """

    return linear_kernel(X,Z)**degree


def polynomial_kernel(X, Z=None, degree=2, gamma=1, coef0=1):
    """computes the polynomial kernel between the samples matrices *X* and *Z*.
    The kernel is defined as k(x,z) = (gamma <x,z> + coef0)**degree
    
    Parameters
    ----------
    X: torch tensor of shape (n_samples_1, n_features)
    Z: torch tensor of shape (n_samples_2, n_features)

    Returns
    -------
    K: torch tensor of shape (n_samples_1, n_samples_2),
        the kernel matrix.
    """

    return (gamma * linear_kernel(X,Z) + coef0)**degree


def euclidean_distances(X, Z=None):
    """computes the pairwise euclidean distances between the samples matrices *X* and *Z*.
    
    Parameters
    ----------
    X: torch tensor of shape (n_samples_1, n_features)
    Z: torch tensor of shape (n_samples_2, n_features)

    Returns
    -------
    D: torch tensor of shape (n_samples_1, n_samples_2),
        the distances matrix.
    """
    
    X, Z = check_pairwise_X_Z(X, Z)
    return torch.cdist(X, Z)


def rbf_kernel(X, Z=None, gamma=1):
    """computes the rbf kernel between the samples matrices *X* and *Z*.
    The kernel is defined as k(x,z) = exp(-gamma ||x-z||^2)
    
    Parameters
    ----------
    X: torch tensor of shape (n_samples_1, n_features)
    Z: torch tensor of shape (n_samples_2, n_features)

    Returns
    -------
    K: torch tensor of shape (n_samples_1, n_samples_2),
        the kernel matrix.
    """

    D = euclidean_distances(X, Z)
    return  torch.exp(-gamma * D**2)

# -*- coding: latin-1 -*-

"""
@author: Ivano Lauriola
@email: ivano.lauriola@phd.unipd.it, ivanolauriola@gmail.com

This file is part of MKLpy: a scikit-compliant framework for Multiple Kernel Learning
This file is distributed with the GNU General Public License v3 <http://www.gnu.org/licenses/>.  

"""

from .exceptions import SquaredKernelError, InvalidKernelsListError, BinaryProblemError
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, balanced_accuracy_score
import torch, numpy as np


def check_X(X):
    if type(X) != torch.Tensor:
        X = torch.tensor(X)
    if len(X.size()) != 2:
        raise ValueError('Wrong shape', X.size())
    X = X.type(torch.float64)
    return X


def check_pairwise_X_Z(X,Z):
    '''checks if X and Z are two broadcastable matrices'''
    Z = X if (Z is None) or (Z is X) else Z
    X = check_X(X)
    Z = check_X(Z)
    if X.size()[1] != Z.size()[1]:
        raise ValueError('X and Z have different features')
    return X, Z


def check_K(K):
    '''checks if a kernel matrix K is squared'''
    K = check_X(K)
    if K.size()[0] != K.size()[1] :
        raise SquaredKernelError(K.size())
    return K


def check_K_Y(K,Y, binary=False):
    '''check if a kernel matrix K and labels vector Y are aligned'''
    K = check_K(K)
    if K.size()[0] != len(Y):
        raise ValueError('The kernel matrix and the labels vector are not aligned')
    if type(Y) != torch.Tensor:
        Y = torch.tensor(Y)
    c = len(Y.unique())
    if binary and c != 2:
        raise BinaryProblemError(c)
    return K, Y


def check_KL(KL):
    '''check if KL is a kernels list'''
    if not hasattr(KL,'__len__'):
        raise InvalidKernelsListError()
    return KL


def check_KL_Y(KL, Y):
    '''check if KL is a kernels list'''
    KL = check_KL(KL)
    if 'Generator' not in type(KL).__name__:
        _, Y = check_K_Y(KL[0], Y)
    return KL, Y


def get_scorer(score, return_direction=False):
    score = score.lower()
    if score == 'roc_auc':
        scorer, f = roc_auc_score, 'decision_function'
    elif score == 'accuracy':
        scorer, f = accuracy_score, 'predict'
    elif score == 'f1_score':
        scorer, f = f1_score, 'predict'
    elif score == 'precision_score':
        scorer, f = precision_score, 'predict'
    elif score == 'recall_score':
        scorer, f = recall_score, 'predict'
    elif score == 'balanced_accuracy_score':
        scorer, f = balanced_accuracy_score, 'predict'
    else:
        raise ValueError('%s is not a valid metric. Valid metrics are \'roc_auc\', \'accuracy\', \'f1_score\', \'precision_score\', \'recall_score\', or \'balanced_accuracy_score\'.' % score)
    if return_direction:
        return scorer, f, np.greater
    else :
        return scorer, f
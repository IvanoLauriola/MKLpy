# -*- coding: latin-1 -*-
 
from sklearn.utils import check_array
import numpy as np
from cvxopt import matrix
 
#controlla se X è effettivamente una lista di kernel
def check_kernel_list(X):
     
    if not hasattr(X,'__len__'):
        raise TypeError("list of kernels must be array-like")
     
    if type(X) == list:
        X = np.array(X)
    elif X.__class__.__name__ == 'kernel_list':
        X = X
    elif type(X) != np.array:
        X = np.array(list(X))

    #return X

    if len(X.shape) == 2:
        raise TypeError("Expected a list of kernels, found a matrix")
    if len(X.shape) != 3:
        raise TypeError("Expected a list of kernels, found unknown")
    a = X[0]
    check_array(a, accept_sparse='csr', dtype=np.float64, order="C")
    
    if a.shape != (X.shape[1],X.shape[2]):
        raise TypeError("Incompatible dimensions")
    '''
    if X[0].shape[1] != X.shape[1]:
        print X[0].shape,' ',X.shape
        raise TypeError("Incompatible dimensions")
    if len(X) > 1 and X[0].shape != X[1].shape:
        raise TypeError("Incompatible dimensions")
    '''
    return X

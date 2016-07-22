import numpy as np
from metrics.pairwise import ideal_kernel, identity_kernel
from regularization import kernel_centering

'''SOLO OPERAZIONI CHE PROCESSANO E RESTITUISCONO UN KERNEL'''
'''TRACE() DOVE CAVOLO LO METTO?'''



''' return the summation of the kernels in list '''
def summation (k_list, weights = None):
    k = np.zeros(k_list[0].shape, dtype = np.double)
    if weights == None:
        weights = np.ones(len(k_list), dtype = np.double)
    for w,ker in zip(weights, k_list):
        k += w * ker
    return k


def multiplication (k_list, weights = None):
    k = np.ones((k_list.shape[1],k_list.shape[2]), dtype = np.float64)
    if weights == None:
        weights = np.ones(k_list.shape[0], dtype = np.float64)
    for w,ker in zip(weights, k_list):
        k *= w * ker
    return k


def mean (k_list, weights = None):
    if weights == None:
        weights = np.ones(k_list.shape[0], dtype = np.float64)
    k = summation(k_list, weights)
    k /= np.sum(weights)
    return k




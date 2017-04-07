"""
.. codeauthor:: Ivano Lauriola <ivanolauriola@gmail.com>

==============================
Check regularization functions
==============================

.. currentmodule:: MKLpy.test.check_regularization

tests over MKLpy.regularization

The following is a complete list of tests performerd:
* 

"""


from numpy.testing import assert_array_equal, assert_array_almost_equal, assert_equal
from MKLpy.lists import HPK_generator
from sklearn.datasets import load_iris
import numpy as np
import sys
from MKLpy.regularization import *

data = load_iris()
Xi,_ = data.data,data.target
n = Xi.shape[0]

Xi=normalization(Xi)


def check_normalization():
    assert np.min(Xi) == 0, "Normalization fail"
    L = np.dot(Xi,Xi.T)
    assert np.max(Xi) == 1, "Normalization fail"
    assert np.min(Xi) == 0, "Normalization fail"
    #check diagonal
    assert_array_almost_equal(np.ones(n),[L[i,i] for i in range(n)])
    return


def check_kernel_normalization():
    return


def check_tracenorm():
    return


def check_rescale():
    return


def check_rescale_01():
    return


def check_centering():
    return


def check_kernel_centering():
    return


def check_edit():
    return
    

def all_check():
    yield check_normalization
    yield check_kernel_normalization
    yield check_tracenorm
    yield check_rescale
    yield check_rescale_01
    yield check_centering
    yield check_kernel_centering
    yield check_edit


'''main'''
def check_regularization():
    for check in all_check():
        check()



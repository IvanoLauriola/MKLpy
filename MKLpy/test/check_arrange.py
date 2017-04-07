"""
.. codeauthor:: Ivano Lauriola <ivanolauriola@gmail.com>

=======================
Check arrange functions
=======================

.. currentmodule:: MKLpy.test.check_arrange

tests over MKLpy.arrange

The following is a complete list of tests performerd:
* check if summation works with a list of kernels
* check if summation works with a ndarray of kernels
* check if summation works with a kernel_list
* check if multiplication works with a list of kernels
* check if multiplication works with a ndarray of kernels
* check if multiplication works with a kernel_list
* check if average works with a list of kernels
* check if average works with a ndarray of kernels
* check if average works with a kernel_list

Note: all checks are performed using 1 and 2 (train and test) kernel matrices.
"""


from numpy.testing import assert_array_equal, assert_array_almost_equal, assert_equal
from MKLpy.lists import HPK_generator
from sklearn.datasets import load_iris
import numpy as np
import sys
from MKLpy.arrange import *
from sklearn.metrics.pairwise import polynomial_kernel as pk
from sklearn.model_selection import train_test_split

data = load_iris()
X,Y = data.data,data.target
Xtr,Xte,Ytr,Yte = train_test_split(X,Y,test_size = .3)

K_list  = [pk(Xtr,degree=i,coef0=0,gamma=1) for i in range(1,6)]
K_array = np.array(K_list)
K_klist = HPK_generator(Xtr).make_a_list(5)

K_list_te  = [pk(Xte,Xtr,degree=i,coef0=0,gamma=1) for i in range(1,6)]
K_array_te = np.array(K_list_te)
K_klist_te = HPK_generator(Xtr,Xte).make_a_list(5)

def check_summation():
    #check summation with 1 samples matrix
    truesum = K_list[0]+K_list[1]+K_list[2]+K_list[3]+K_list[4]
    sum_list  = summation(K_list)
    sum_array = summation(K_array)
    sum_klist = summation(K_klist)
    assert_array_equal(sum_list,truesum)
    assert_array_equal(sum_list,sum_array)
    assert_array_equal(sum_list,sum_klist)
    #check summation with 2 samples matrix
    truesum_te = K_list_te[0]+K_list_te[1]+K_list_te[2]+K_list_te[3]+K_list_te[4]
    sum_list_te  = summation(K_list_te)
    sum_array_te = summation(K_array_te)
    sum_klist_te = summation(K_klist_te)
    assert_array_equal(sum_list_te,truesum_te)
    assert_array_equal(sum_list_te,sum_array_te)
    assert_array_equal(sum_list_te,sum_klist_te)
    return

def check_multiplication():
    #check multiplication with 1 samples matrix
    truemul = K_list[0]*K_list[1]*K_list[2]*K_list[3]*K_list[4]
    mul_list  = multiplication(K_list)
    mul_array = multiplication(K_array)
    mul_klist = multiplication(K_klist)
    assert_array_equal(mul_list,truemul)
    assert_array_equal(mul_list,mul_array)
    assert_array_equal(mul_list,mul_klist)
    #check multiplication with 2 samples matrix
    truemul_te = K_list_te[0]*K_list_te[1]*K_list_te[2]*K_list_te[3]*K_list_te[4]
    mul_list_te  = multiplication(K_list_te)
    mul_array_te = multiplication(K_array_te)
    mul_klist_te = multiplication(K_klist_te)
    assert_array_equal(mul_list_te,truemul_te)
    assert_array_equal(mul_list_te,mul_array_te)
    assert_array_equal(mul_list_te,mul_klist_te)
    return

def check_average():
    #check average with 1 samples matrix
    trueaverage = (K_list[0]+K_list[1]+K_list[2]+K_list[3]+K_list[4])/5.0
    average_list  = average(K_list)
    average_array = average(K_array)
    average_klist = average(K_klist)
    assert_array_equal(average_list,trueaverage)
    assert_array_equal(average_list,average_array)
    assert_array_equal(average_list,average_klist)
    #check average with 2 samples matrix
    trueaverage_te = (K_list_te[0]+K_list_te[1]+K_list_te[2]+K_list_te[3]+K_list_te[4])/5.0
    average_list_te  = average(K_list_te)
    average_array_te = average(K_array_te)
    average_klist_te = average(K_klist_te)
    assert_array_equal(average_list_te,trueaverage_te)
    assert_array_equal(average_list_te,average_array_te)
    assert_array_equal(average_list_te,average_klist_te)
    return



def all_check():
    yield check_summation
    yield check_multiplication
    yield check_average


'''main'''
def check_arrange():
    for check in all_check():
        check()


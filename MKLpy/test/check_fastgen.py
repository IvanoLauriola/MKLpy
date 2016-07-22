from numpy.testing import assert_array_equal, assert_array_almost_equal, assert_equal
from MKLpy.lists import fast_HPK
from MKLpy.lists import HPK_generator
from sklearn.datasets import load_iris
import numpy as np
import time
import sys
from MKLpy.regularization import normalization

data = load_iris()
X,Y = data.data,data.target
X=normalization(X)

def check_HPK_numerical():
    kf = fast_HPK(X,l=20)
    kl = HPK_generator(X).make_a_list(20).to_array()
    for i in [0,1,2,6,10,15,19]:
        assert_array_almost_equal(kf[i],kl[i])
    return



def all_check():
    yield check_HPK_numerical




'''main'''
def check_fastgen():
    for check in all_check():
        check()

check_fastgen()

from numpy.testing import assert_array_equal, assert_array_almost_equal, assert_equal
from MKLpy.lists import HPK_generator
from sklearn.datasets import load_iris
import numpy as np
import sys
from MKLpy.regularization import normalization

data = load_iris()
X,Y = data.data,data.target
#X=normalization(X)



def all_check():
    yield nome




'''main'''
def check_metrics():
    for check in all_check():
        check()

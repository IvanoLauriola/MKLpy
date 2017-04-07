"""
.. codeauthor:: Ivano Lauriola <ivanolauriola@gmail.com>

==============================
Check regularization functions
==============================

.. currentmodule:: MKLpy.test.check_search

tests over MKLpy.model_selection, that comprises searching strategies, cross_validation ad splits

The following is a complete list of tests performerd:
* 

"""

import sys
import numpy as np
from numpy.testing    import assert_array_equal, assert_array_almost_equal, assert_equal
from sklearn.datasets import load_iris,load_digits,load_breast_cancer
from sklearn.model_selection import KFold, ShuffleSplit
from MKLpy.lists      import HPK_generator
from MKLpy.model_selection import *
from MKLpy.algorithms import EasyMKL

data = load_breast_cancer()#digits()#iris()
X,Y = data.data,data.target
Y = np.array([1 if _y==Y[1] else -1 for _y in Y])
#Y = np.array([1 if _y < 5 else -1 for _y in Y])
n = X.shape[0]
y = np.array(range(n))
KL = HPK_generator(X).make_a_list(10).to_array()


def check_cv3():
    return
    cv = KFold(3).split(Y)
    for train,test in cv:
        tr,te = cv3(train,test,10)
        assert KL[tr].shape == (10,100,100)
        assert KL[te].shape == (10,50,100)
    
def check_train_test_split():
    #check shape with train_size (%)
    Ktr,Kte,Ytr,Yte = train_test_split(KL,Y,train_size=.3)
    assert Ktr.shape == (10,n*0.3,n*0.3), 'Shape error'
    assert Kte.shape == (10,n*0.7,n*0.3), 'Shape error'
    assert Ytr.shape[0] == n*0.3
    assert Yte.shape[0] == n*0.7
    #check shape with test_size (%)
    Ktr,Kte,Ytr,Yte = train_test_split(KL,Y,test_size=.3)
    assert Ktr.shape == (10,n*0.7,n*0.7), 'Shape error'
    assert Kte.shape == (10,n*0.3,n*0.7), 'Shape error'
    assert Ytr.shape[0] == n*0.7
    assert Yte.shape[0] == n*0.3
    #check shape with train_size
    Ktr,Kte,Ytr,Yte = train_test_split(KL,Y,train_size=100)
    assert Ktr.shape == (10,100,100), 'Shape error'
    assert Kte.shape == (10,50,100), 'Shape error'
    assert Ytr.shape[0] == 100
    assert Yte.shape[0] == 50
    #check samples and indices
    Ktr,Kte,Ytr,Yte = train_test_split(KL,y)
    assert len(y) == len(np.unique(np.concatenate([Ytr,Yte])))
    assert KL[2,Ytr[3],Ytr[4]] == Ktr[2,3,4]
    assert KL[4,Ytr[9],Ytr[7]] == Ktr[4,9,7]
    

def check_cross_validation():
    scores = cross_val_score(KL,Y,EasyMKL(lam=0.1,kernel='precomputed'),n_folds=3)
    assert len(scores) == 3
    pass

def check_GridSearchCV():
    pass

def check_check_cv():
    pass

def check_check_scoring():
    pass




    
def all_check():
    yield check_cv3
    #yield check_train_test_split
    yield check_cross_validation
    yield check_GridSearchCV
    yield check_check_cv
    yield check_check_scoring


'''main'''
def check_search():
    for check in all_check():
        check()


check_search()

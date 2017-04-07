from cvxopt import matrix, solvers, spdiag
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel, linear_kernel
from sklearn.utils import check_array, check_consistent_length, check_random_state
from sklearn.utils import column_or_1d, check_X_y
from sklearn.utils import validation
from sklearn.utils import check_array
from sklearn.exceptions import NotFittedError
from sklearn.svm import SVC     #base learner
from MKLpy.arrange import summation


'''
combine kernels using such heuristics defined as input.
Ginev
 - a kernels list
 - a function that define an initial weights for each kernel
     this function has a kernel and the lables as inputs and it returns an int value
 - a function that modify the final weights
     returns a weights vector w where \|w\|_1 = 1
     this function can have at least 1 params (with default value)
 - a functional form able to combine kernels
 - a base learner
performs
 - the kernels combination
'''
class HeuristicMKLClassifier :

    def __init__(self, f, estimator = SVC(),
                 post_process = None,#base_transformation,
                 #lam = 0.0,
                 functional_form = summation):
        '''
        f : a function, f: kernel X labels -> int, returns a score given a kernel
        lam : a value between [0,1] that represents the variance in weights
        estimator : the base learner
        distrib : a mechanism to post-process weights
            - if 'real' -> the weights are exactly the results of f
            - else -> the weights are post-processed according to lam
        non_negative : if True, only positive weights are calculated
        post_process : a function where the input is a weights vector and perform a transformation
        '''
        self.f   = f
        self.lam = lam
        self.base = estimator
        self.post_process = post_process
        self.functional_form = functional_form

    def fit(K,Y):
        Y = [1 if _y == Y[0] else -1 for _y in Y]
        self.n_kernels  = K.shape[0]
        self.ker_matrix = arrange_kernel(K,Y)
        self.clf = self.base.__class__(**self.base.get_params())
        self.clf.set_params({'kernel':'precomputed'})
        self.clf = self.clf.fit(self.ker_matrix,Y)

    def arrange_kernel(K,Y):
        weights = self.post_process([self.f(_k,Y) for _k in K])
        return self.functional_form(K,self.weights)

    def decision_funcion(K):
        return self.clf.decision_function(self.functional_form(K,self.weights))

    def predict(K):
        return self.clf.predict(self.functional_form(K,self.weights))






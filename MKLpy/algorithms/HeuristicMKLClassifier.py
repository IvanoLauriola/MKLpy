from cvxopt import matrix, solvers
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel, linear_kernel
from sklearn.utils import check_array, check_consistent_length, check_random_state
from sklearn.utils import column_or_1d, check_X_y
from sklearn.utils import validation
from sklearn.utils import check_array
#from sklearn.exceptions import NotFittedError
from sklearn.svm import SVC     #base learner

from MKLpy.arrange import summation

class HeuristicMKLClassifier :

    def __init__(self, f, beta0=0, beta1=1, beta2=1, C=1, coef0=0,gamma=1, degree=2):
        self.f = f
        self.beta0 = beta0
        self.beta1 = beta1
        self.beta2 = beta2
        self.coef0 = coef0      #
        self.gamma = gamma      #   per la generazione automatica di kernel
        self.degree = degree    #
        self.C = C

    def fit(K,Y):
        ker_matrix = arrange_kernel(K,Y)
        self.ker_matrix = ker_matrix
        clf = SVC(C=self.C,kernel='precomputed').fit(ker_matrix,Y)
        self.clf = clf
        self.Y = Y
        return self
        

    def arrange_kernel(K,Y):
        weights = []
        for k in K:
            weights.append(self.beta0 + (self.beta1 * f(k,Y))** self.beta2)
        self.weights = np.linalg.norm(weights,1)
        return summation(K,self.weights)

    def decision_funcion(K):
        return self.clf.decision_function(summation(K,self.weights))

    def predict(K):
        return self.clf.predict(summation(K,self.weights))

    def predict_proba(K):
        return self.clf.predict_proba(summation(K,self.weights))






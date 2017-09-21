# -*- coding: latin-1 -*-

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
from base import MKL
from MKLpy.multiclass import OneVsOneMKLClassifier as ovoMKL, OneVsOneMKLClassifier as ovaMKL   #ATTENZIONE DUE OVO
from MKLpy.utils.exceptions import ArrangeMulticlassError
from MKLpy.algorithms import KOMD
from MKLpy.lists import HPK_generator
 
from cvxopt import matrix, spdiag, solvers
import numpy as np
 
from MKLpy.arrange import summation
 
 
class EasyRKL(BaseEstimator, ClassifierMixin, MKL):

    def __init__(self, estimator=SVC(), lam=0.0, generator=HPK_generator(n=10), multiclass_strategy='ova', max_iter=100, verbose=False):
        super(self.__class__, self).__init__(estimator=estimator, generator=generator, multiclass_strategy=multiclass_strategy, how_to=summation, max_iter=max_iter, verbose=verbose)
        self.lam = lam

        
    def _arrange_kernel(self):
        Y = [1 if y==self.classes_[1] else -1 for y in self.Y]
        n_sample = len(self.Y)
        ker_matrix = matrix(summation(self.KL))
        YY = spdiag(Y)
        KLL = (1.0-self.lam)*YY*ker_matrix*YY
        LID = spdiag([self.lam]*n_sample)
        Q = 2*(KLL+LID)
        p = matrix([0.0]*n_sample)
        G = -spdiag([1.0]*n_sample)
        h = matrix([0.0]*n_sample,(n_sample,1))
        A = matrix([[1.0 if lab==+1 else 0 for lab in Y],[1.0 if lab2==-1 else 0 for lab2 in Y]]).T
        b = matrix([[1.0],[1.0]],(2,1))
         
        solvers.options['show_progress'] = False
        solvers.options['maxiters'] = self.max_iter
        sol = solvers.qp(Q,p,G,h,A,b)
        gamma = sol['x']

        yg = gamma.T * YY
        weights_margin = [(yg*matrix(K)*yg.T)[0] for K in self.KL]

        P = 2 * ker_matrix
        p = -matrix([ker_matrix[i,i] for i in range(n_sample)])
        G = -spdiag([1.0] * n_sample)
        h = matrix([0.0] * n_sample)
        A = matrix([1.0] * n_sample).T
        b = matrix([1.0])
        solvers.options['show_progress']=False
        sol = solvers.qp(P,p,G,h,A,b)
        alpha = sol['x']
        weights_radius = [(alpha.T * matrix(K) * alpha)[0] for K in self.KL]

        
        weights = [a/b for (a,b) in zip(weights_radius,weights_margin)]
        norm2 = sum([w for w in weights])
        self.weights = np.array([w / norm2 for w in weights])
        ker_matrix = summation(self.KL, self.weights)
        self.ker_matrix = ker_matrix
        return ker_matrix

 
    def get_params(self, deep=True):
        # this estimator has parameters:
        return {"lam": self.lam,
                "generator": self.generator, "n_kernels": self.n_kernels, "max_iter":self.max_iter,
                "verbose":self.verbose, "multiclass_strategy":self.multiclass_strategy,
                'estimator':self.estimator}


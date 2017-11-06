# -*- coding: latin-1 -*-
"""
"""
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
from base import MKL
from MKLpy.multiclass import OneVsOneMKLClassifier as ovoMKL, OneVsOneMKLClassifier as ovaMKL   #ATTENZIONE DUE OVO
from MKLpy.utils.exceptions import ArrangeMulticlassError
from MKLpy.lists import HPK_generator
from MKLpy.metrics import margin
from MKLpy.metrics import spectral_ratio,frobenius
import sys
from cvxopt import matrix, spdiag, solvers
import numpy as np
 
from MKLpy.arrange import summation
from MKLpy.algorithms import KOMD



class MOME(BaseEstimator, ClassifierMixin, MKL):


    def __init__(self, estimator=SVC(), lam=0, C=0.1, step=0.002, generator=HPK_generator(n=10), multiclass_strategy='ova', max_iter=1000, tol=1e-7, verbose=False):
        super(self.__class__, self).__init__(estimator=estimator, generator=generator, multiclass_strategy=multiclass_strategy, how_to=summation, max_iter=max_iter, verbose=verbose)
        self.step = step
        self.tol = tol
        self.lam = lam
        self.C = C



    def _arrange_kernel(self):
        Y = [1 if y==self.classes_[1] else -1 for y in self.Y]
        nn = len(Y)
        nk = self.n_kernels
        YY = spdiag(Y)
        beta = [0.0] * nk
        mu = np.exp(beta)
        mu /= mu.sum()
        
        #actual_weights = eta[:]
        actual_ratio = None
        Q = np.array([[np.dot(self.KL[r].ravel(),self.KL[s].ravel()) for r in range(nk)] for s in range(nk)])
        Q /= np.sum([frobenius(K)**2 for K in self.KL])

        self.sr,self.margin = [],[]
        self.obj = []
        _beta,_mu = None,None
        cstep = self.step
        I = np.diag(np.ones(nn))
        for i in xrange(self.max_iter):
            Kc = summation(self.KL,mu)
            #trovo i gamma
            clf = KOMD(kernel='precomputed',lam=self.lam).fit(Kc,Y)
            gamma = clf.gamma
            _margin = (gamma.T * YY * matrix(Kc) * YY * gamma)[0]
            #m = (gamma.T * YY * matrix(Kc) * YY * gamma)[0]
            grad = np.array([(self.C * np.dot(Q[r],mu) + (gamma.T * YY * matrix(self.KL[r]) * YY * gamma)[0]) \
                    * mu[r] * (1- mu[r]) \
                      for r in range(nk)])
            _beta = beta + cstep * grad
            _mu = np.exp(_beta)
            _mu /= _mu.sum()

            _obj = _margin+(self.C/2)* np.dot(_mu,np.dot(_mu,Q))
            if self.obj and _obj<self.obj[-1]:  # nel caso di peggioramento
                cstep /= 2.0
                if cstep < 0.00001: break
            else : 
                self.obj.append(_obj)
                self.margin.append(_margin)
                mu = _mu
                beta = _beta
            if _margin < 1e-3:
                break


        self._steps = i+1
        #print 'steps', self._steps, 'lam',self.lam,'margin',self.margin[-1]
        self.weights = np.array(mu)
        self.ker_matrix = summation(self.KL,self.weights)
        return self.ker_matrix

 
    def get_params(self, deep=True):
        # this estimator has parameters:
        return {"step": self.step,
                "tol": self.tol,
                "C":self.C,
                "lam":self.lam,
                "generator": self.generator, "max_iter":self.max_iter,
                "verbose":self.verbose, "multiclass_strategy":self.multiclass_strategy,
                "estimator":self.estimator}


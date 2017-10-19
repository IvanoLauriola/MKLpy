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


    def __init__(self, estimator=SVC(), lam=0, step=0.01, generator=HPK_generator(n=10), multiclass_strategy='ova', max_iter=500, tol=1e-7, verbose=False):
        super(self.__class__, self).__init__(estimator=estimator, generator=generator, multiclass_strategy=multiclass_strategy, how_to=summation, max_iter=max_iter, verbose=verbose)
        self.step = step
        self.tol = tol
        self.lam = lam



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

        self.sr,self.margin = [],[]
        cstep = self.step
        I = np.diag(np.ones(nn))
        for i in xrange(self.max_iter):
        	Kc = summation(self.KL,mu)
        	#self.sr.append(spectral_ratio(Kc))
        	#Kcc = (1-self.lam) * Kc + self.lam * np.diag(np.ones(nn))
        	self.sr.append(np.sum(Kc**2))
        	try:
        		clf = KOMD(lam=self.lam,kernel='precomputed').fit(Kc,Y)
        		gamma = clf.gamma
        		#m = margin(Kcc,Y)
        		m = (gamma.T * YY * matrix(Kc) * YY * gamma)[0]

        		self.margin.append(m)
        	except:
        		break;
        	if m < 0.01:
        		break;
        	else:
        		pass
        		#print m
        		#sys.stdout.flush()

        	#margin maximization
        	#if len(self.margin)>1 and self.margin[-1] < self.margin[-2]:
        	#	break;
        	#end margin mazimization

        	gstep = np.array([2 * np.dot(Q[r],mu) * mu[r] * (1 - mu[r]) for r in range(nk)])
        	beta += cstep * gstep
        	#eta /= eta.sum()
        	mu = np.exp(beta)
        	mu /= mu.sum()


        self._steps = i+1
        print 'steps', self._steps, 'lam',self.lam,'margin',self.margin[-1]
        
        self.weights = np.array(mu)
        self.ker_matrix = summation(self.KL,self.weights)
        return self.ker_matrix

 
    def get_params(self, deep=True):
        # this estimator has parameters:
        return {"step": self.step,
                "tol": self.tol,
                "generator": self.generator, "n_kernels": self.n_kernels, "max_iter":self.max_iter,
                "verbose":self.verbose, "multiclass_strategy":self.multiclass_strategy,
                'estimator':self.estimator}


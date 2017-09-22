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




class MOME(BaseEstimator, ClassifierMixin, MKL):


    def __init__(self, estimator=SVC(), step=0.01, generator=HPK_generator(n=10), multiclass_strategy='ova', max_iter=500, tol=1e-7, verbose=False):
        super(self.__class__, self).__init__(estimator=estimator, generator=generator, multiclass_strategy=multiclass_strategy, how_to=summation, max_iter=max_iter, verbose=verbose)
        self.step = step
        self.tol = tol


    def _arrange_kernel(self):
        Y = [1 if y==self.classes_[1] else -1 for y in self.Y]
        nn = len(Y)
        nk = self.n_kernels
        YY = spdiag(Y)
        eta = [1.0 / nk] * nk
        
        actual_weights = eta[:]
        actual_ratio = None
        Q = np.array([[np.dot(self.KL[r].ravel(),self.KL[s].ravel()) for r in range(nk)] for s in range(nk)])

        self.sr,self.margin = [],[]
        cstep = self.step
        for i in xrange(self.max_iter):
        	Kc = summation(self.KL,eta)
        	#self.sr.append(spectral_ratio(Kc))
        	self.sr.append(np.sum(Kc**2))
        	try:
        		m = margin(Kc,Y)
        		self.margin.append(m)
        	except:
        		break;
        	if m < 0.0001:
        		break;
        	else:
        		pass
        		#print m
        		#sys.stdout.flush()
        	gstep = np.array([2 * np.dot(Q[r],eta) * eta[r] * (1 - eta[r]) for r in range(nk)])
        	eta += cstep * gstep
        	eta /= eta.sum()


        self._steps = i+1
        print 'steps', self._steps
        
        self.weights = np.array(eta)
        self.ker_matrix = summation(self.KL,self.weights)
        return self.ker_matrix

 
    def get_params(self, deep=True):
        # this estimator has parameters:
        return {"step": self.step,
                "tol": self.tol,
                "generator": self.generator, "n_kernels": self.n_kernels, "max_iter":self.max_iter,
                "verbose":self.verbose, "multiclass_strategy":self.multiclass_strategy,
                'estimator':self.estimator}


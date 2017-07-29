# -*- coding: latin-1 -*-
"""
"""
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
from base import MKL
from MKLpy.multiclass import OneVsOneMKLClassifier as ovoMKL, OneVsOneMKLClassifier as ovaMKL   #ATTENZIONE DUE OVO
from MKLpy.utils.exceptions import ArrangeMulticlassError
from MKLpy.lists import HPK_generator
 
from cvxopt import matrix, spdiag, solvers
import numpy as np
 
from MKLpy.arrange import summation
 



def radius(K):   #modificati per ritornare anche x
    n = K.shape[0]
    P = 2 * matrix(K)
    p = -matrix([K[i,i] for i in range(n)])
    G = -spdiag([1.0] * n)
    h = matrix([0.0] * n)
    A = matrix([1.0] * n).T
    b = matrix([1.0])
    solvers.options['show_progress']=False
    sol = solvers.qp(P,p,G,h,A,b)
    #return abs(sol['primal objective'])
    return np.sqrt(abs(sol['primal objective'])),sol['x']

def margin(K,Y):
    n = len(Y)
    YY = spdiag(list(Y))
    P = 2*(YY*matrix(K)*YY)
    p = matrix([0.0]*n)
    G = -spdiag([1.0]*n)
    h = matrix([0.0]*n)
    A = matrix([[1.0 if Y[i]==+1 else 0 for i in range(n)],
                [1.0 if Y[j]==-1 else 0 for j in range(n)]]).T
    b = matrix([[1.0],[1.0]],(2,1))
    solvers.options['show_progress']=False
    sol = solvers.qp(P,p,G,h,A,b)
    return np.sqrt(sol['primal objective']),sol['x']







class RMGD(BaseEstimator, ClassifierMixin, MKL):


    def __init__(self, estimator=SVC(), step=1, generator=HPK_generator(n=10), multiclass_strategy='ova', max_iter=500, tol=1e-7, verbose=False):
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
        Kc = summation(self.KL,eta)
        _r, alpha = radius(Kc)
        _m, gamma = margin(Kc,Y)

        self._ratios = []

        cstep = self.step
        for i in xrange(self.max_iter):
            #print actual_ratio

            a = np.array([1.0-(alpha.T*matrix(K)*alpha)[0] for K in self.KL])
            b = np.array([(gamma.T*YY*matrix(K)*YY*gamma)[0] for K in self.KL])            
            den = [np.dot(eta,b)**2]*nk
            num = [sum([eta[s]*(a[s]*b[r]-a[r]*b[s])  for s in range(nk)])  for r in range(nk)]

            eta = [cstep * (num[k]/den[k]) + eta[k] for k in range(nk)]
            eta = [max(0,v) for v in eta]
            eta = np.array(eta)/sum(eta)


            Kc = summation(self.KL,eta)
            _r, alpha = radius(Kc)
            _m, gamma = margin(Kc,Y)

            new_ratio = _r**2/_m**2
            if actual_ratio and abs(new_ratio - actual_ratio)/nn < self.tol:
                #completato
                #print i,'tol'
                self._ratios.append(new_ratio)
                actual_weights = eta
                #break;             #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            elif new_ratio <= actual_ratio or not actual_ratio:
                #tutto in regola
                actual_ratio = new_ratio
                self._ratios.append(actual_ratio)
                #print i,'update',actual_ratio
                actual_weights = eta
            else:
                #print i,'revert'
                #supero il minimo
                eta = actual_weights
                cstep /= 1.50
                continue
        self._steps = i+1
        
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


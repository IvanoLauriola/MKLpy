# -*- coding: latin-1 -*-
"""
"""
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
from MKLpy.algorithms.base import MKL
from MKLpy.multiclass import OneVsOneMKLClassifier as ovoMKL, OneVsOneMKLClassifier as ovaMKL   #ATTENZIONE DUE OVO
from MKLpy.utils.exceptions import ArrangeMulticlassError
from MKLpy.lists import HPK_generator
 
from cvxopt import matrix, spdiag, solvers
import numpy as np
import time
from MKLpy.arrange import summation
 



def radius(K,init_sol=None): 
    #init_sol=None
    n = K.shape[0]
    P = 2 * matrix(K)
    p = -matrix([K[i,i] for i in range(n)])
    G = -spdiag([1.0] * n)
    h = matrix([0.0] * n)
    A = matrix([1.0] * n).T
    b = matrix([1.0])
    solvers.options['show_progress']=False
    sol = solvers.qp(P,p,G,h,A,b,initvals=init_sol)# if init_sol else solvers.qp(P,p,G,h,A,b)
    return np.sqrt(abs(sol['primal objective'])),sol['x'],sol


def margin(K,Y,init_sol=None):
    #init_sol=None
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
    sol = solvers.qp(P,p,G,h,A,b,initvals=init_sol) # if init_sol else solvers.qp(P,p,G,h,A,b)
    #raw_input()
    return np.sqrt(sol['primal objective']),sol['x'],sol







class GRAM(BaseEstimator, ClassifierMixin, MKL):

    def __init__(self, estimator=SVC(), step=1, generator=HPK_generator(n=10), multiclass_strategy='ova', max_iter=1000, tol=1e-9, verbose=False):
        super(self.__class__, self).__init__(estimator=estimator, generator=generator, multiclass_strategy=multiclass_strategy, how_to=summation, max_iter=max_iter, verbose=verbose)
        self.step = step
        self.tol = tol


    def _arrange_kernel(self):
        Y = [1 if y==self.classes_[1] else -1 for y in self.Y]
        nn = len(Y)
        nk = self.n_kernels
        YY = spdiag(Y)
        beta = [0.0] * nk
        mu = np.exp(beta)
        mu /= mu.sum()
        
        Kc = summation(self.KL,mu)
        _r, alpha, _sol_a_old = radius(Kc)
        _m, gamma, _sol_g_old = margin(Kc,Y)
        self._ratios = []

        cstep = self.step
        self._converg = False
        self._steps = 0

        while (not self._converg and (self._steps < self.max_iter)):
            self._steps += 1
            eb = np.exp(beta)

            #calcolo il gradiene
            a = np.array([1.0-(alpha.T*matrix(K)*alpha)[0] for K in self.KL])
            b = np.array([(gamma.T*YY*matrix(K)*YY*gamma)[0] for K in self.KL])            
            den = [np.dot(eb,b)**2]*nk
            #num = [sum([eta[s]*(a[s]*b[r]-a[r]*b[s])  for s in range(nk)])  for r in range(nk)]
            num = [eb[r] * (a[r]*np.dot(eb,b)   -   b[r]*np.dot(eb,a)) for r in range(nk)]

            #calcolo i pesi temporanei
            _beta = [beta[k] - cstep * (num[k]/den[k]) for k in range(nk)]
            current_mu = np.exp(_beta)/np.exp(_beta).sum()
           
            #testo la nuova soluzione
            try:
              Kc = summation(self.KL,current_mu)
              _r, alpha, _sol_a = radius(Kc,init_sol=_sol_a_old)
              _m, gamma, _sol_g = margin(Kc,Y,init_sol=_sol_g_old)
              _sol_a_old = _sol_a.copy()
              _sol_g_old = _sol_g.copy()
            except :
            	print '### warning at step %d:' % self._steps
            	print 'current weights:', current_mu
            	cstep /= 2.0


            new_ratio = (_r**2/_m**2) / nn
            #caso 1: primo passo o soluzione migliorativa
            if not self._ratios or self._ratios[-1] > new_ratio:
            	#aggiorno lo stato
            	#print 'soluzione migliorativa'
            	beta = _beta
            	mu = current_mu
            	self._ratios.append(new_ratio)

            #caso 2: soluzione peggiorativa
            elif self._ratios[-1] <= new_ratio:
            	cstep /= 2.0

            #controllo sulla convergenza
            if 	cstep <= 1e-10 or 	\
            	( len(self._ratios)>=2 and np.linalg.norm(self._ratios[-1]-self._ratios[-2]) <= 1e-20) :
            	self._converg = True

        
        self.weights = np.array(mu)
        self.ker_matrix = summation(self.KL,self.weights)
        return self.ker_matrix

 
    def get_params(self, deep=True):
        # this estimator has parameters:
        return {"step": self.step,
                "tol": self.tol,
                "generator": self.generator, "max_iter":self.max_iter,
                "verbose":self.verbose, "multiclass_strategy":self.multiclass_strategy,
                'estimator':self.estimator}

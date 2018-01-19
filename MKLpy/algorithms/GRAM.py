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
import time,sys
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
    #solvers.options['feastol']=1
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
    #solvers.options['feastol']=1e-8
    sol = solvers.qp(P,p,G,h,A,b,initvals=init_sol) # if init_sol else solvers.qp(P,p,G,h,A,b)
    return np.sqrt(sol['primal objective']),sol['x'],sol







class GRAM(BaseEstimator, ClassifierMixin, MKL):

    def __init__(self, estimator=SVC(), step=1.0, generator=HPK_generator(n=10), multiclass_strategy='ova', max_iter=1000, tol=1e-9, verbose=False, n_folds=1, random_state=42):
        super(self.__class__, self).__init__(estimator=estimator, generator=generator, multiclass_strategy=multiclass_strategy, how_to=summation, max_iter=max_iter, verbose=verbose)
        self.step = step
        self.tol = tol
        self.n_folds = n_folds
        self.random_state = random_state

    def _arrange_kernel(self):

        Y = [1 if y==self.classes_[1] else -1 for y in self.Y]
        nn = len(Y)
        nk = self.n_kernels

        idx_e = range(nn)
        np.random.seed(self.random_state)
        np.random.shuffle(idx_e)
        splits = [idx_e[i::self.n_folds] for i in range(self.n_folds)]
        alphas = [None] * self.n_folds
        gammas = [None] * self.n_folds
        radiuses = [None] * self.n_folds
        margins  = [None] * self.n_folds

        YY = spdiag(Y)
        beta = [0.0] * nk
        mu = np.exp(np.array(beta)-max(beta))
        mu /= mu.sum()
        
        Kc = summation(self.KL,mu)
        Y = np.array(Y)
        for _i,idx in enumerate(splits):
            _,_,alphas[_i] = radius(Kc[idx][:,idx])
            _,_,gammas[_i] = margin(Kc[idx][:,idx],Y[idx])
        self._ratios = []

        cstep = self.step
        self._converg = False
        self._steps   = 0

        while (not self._converg and (self._steps < self.max_iter)):
            self._steps += 1

            new_ratio = 0.0
            _beta = beta[:]
            current_mu = mu[:]
            for _i,idx in enumerate(splits):

                #calcolo il gradiente
                Ks = Kc[idx][:,idx]
                YYs = YY[idx,idx]
                eb = np.exp(np.array(_beta))

                a = np.array([1.0-(alphas[_i]['x'].T*matrix(K[idx][:,idx])*alphas[_i]['x'])[0] for K in self.KL])
                b = np.array([(gammas[_i]['x'].T*YYs*matrix(K[idx][:,idx])*YYs*gammas[_i]['x'])[0] for K in self.KL])            
                den = [np.dot(eb,b)**2]*nk
                num = [eb[r] * (a[r]*np.dot(eb,b)   -   b[r]*np.dot(eb,a)) for r in range(nk)]

                #calcolo i pesi temporanei
                _beta = [_beta[k] - cstep * (num[k]/den[k]) for k in range(nk)]
                current_mu = np.exp(_beta-max(_beta))
                current_mu /= current_mu.sum()
                #print _i,current_mu
                #testo la nuova soluzione
                try:
                  Kc = summation(self.KL,current_mu)
                  _r, _, alphas[_i] = radius(Kc[idx][:,idx],init_sol=alphas[_i].copy())
                  _m, _, gammas[_i] = margin(Kc[idx][:,idx],Y[idx],init_sol=gammas[_i].copy())
                  radiuses[i] = _r
                  margins[i]  = _m
                except :
                    Kc = summation(self.KL,current_mu)
                    _r, _, alphas[_i] = radius(Kc[idx][:,idx])
                    _m, _, gammas[_i] = margin(Kc[idx][:,idx],Y[idx])
                    radiuses[i] = _r
                    margins[i]  = _m
                    #print '### warning at step %d:' % self._steps
                    #print 'current weights:', current_mu
                    #print "Unexpected error:", sys.exc_info()
                    #current_mu = mu[:]
                    cstep /= 2.0




                #new_ratio += (_r**2/_m**2) / len(idx)
                new_ratio += (radiuses[i]**2/margins[i]**2) / len(idx)

            new_ratio /= self.n_folds
            #print self._steps,new_ratio

            #caso 1: primo passo o soluzione migliorativa
            if not self._ratios or self._ratios[-1] > new_ratio:
                #aggiorno lo stato
                #print 'soluzione migliorativa'
                beta = _beta
                mu = current_mu
                self.weights = np.array([mm for mm in mu])
                self._ratios.append(new_ratio)

            #caso 2: soluzione peggiorativa
            elif self._ratios[-1] <= new_ratio:
                print 'passo peggiorativo, (%f)' % cstep
                cstep /= 2.0

            #controllo sulla convergenza
            if     cstep <= 1e-8 or     \
            	( len(self._ratios)>=2 and abs(self._ratios[-1]-self._ratios[-2]) <= 1e-10 and self._steps>10):
                self._converg = True
                print 'convergenza',cstep

        
        #self.weights = np.array(mu)
        self.ker_matrix = summation(self.KL,self.weights)
        return self.ker_matrix

 
    def get_params(self, deep=True):
        # this estimator has parameters:
        return {"step": self.step,
                "tol": self.tol,
                "generator": self.generator, "max_iter":self.max_iter,
                "verbose":self.verbose, "multiclass_strategy":self.multiclass_strategy,
                'estimator':self.estimator}

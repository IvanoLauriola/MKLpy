# -*- coding: latin-1 -*-
"""
"""

from .AlternateMKL import AlternateMKL, Cache
from .base import Solution
from ..arrange import average, summation
from ..utils.misc import uniform_vector
from sklearn.svm import SVC 
from cvxopt import matrix, spdiag, solvers
import numpy as np
import time,sys 
from ..metrics import ratio



def opt_radius(K, init_sol=None): 
    n = K.shape[0]
    K = matrix(K)
    P = 2 * K
    p = -matrix([K[i,i] for i in range(n)])
    G = -spdiag([1.0] * n)
    h = matrix([0.0] * n)
    A = matrix([1.0] * n).T
    b = matrix([1.0])
    solvers.options['show_progress']=False
    sol = solvers.qp(P,p,G,h,A,b,initvals=init_sol)
    radius2 = (-p.T * sol['x'])[0] - (sol['x'].T * K * sol['x'])[0]
    return sol, radius2


def opt_margin(K, YY, init_sol=None):
    '''optimized margin evaluation'''
    n = K.shape[0]
    P = 2 * (YY * matrix(K) * YY)
    p = matrix([0.0]*n)
    G = -spdiag([1.0]*n)
    h = matrix([0.0]*n)
    A = matrix([[1.0 if YY[i,i]==+1 else 0 for i in range(n)],
                [1.0 if YY[j,j]==-1 else 0 for j in range(n)]]).T
    b = matrix([[1.0],[1.0]],(2,1))
    solvers.options['show_progress']=False
    sol = solvers.qp(P,p,G,h,A,b,initvals=init_sol) 
    margin2 = sol['primal objective']
    return sol, margin2





class GRAM(AlternateMKL):

    def __init__(self, 
        learner=SVC(C=1000), 
        multiclass_strategy='ova', 
        verbose=False,
        max_iter=1000, 
        learning_rate=0.01, 
        tolerance=1e-7, 
        callbacks=[], 
        scheduler=None ):

        super().__init__(
            learner=learner, 
            multiclass_strategy=multiclass_strategy, 
            max_iter=max_iter, 
            verbose=verbose, 
            tolerance=tolerance,
            callbacks=callbacks,
            scheduler=scheduler, 
            direction='min', 
            learning_rate=learning_rate, 
        )
        self.func_form = summation


    def get_params(self, deep=True):
        params = super().get_params()
        #no additional algorithm-specific parameters
        return params


    def initialize_optimization(self):
        YY          = spdiag([1 if y==self.Y[0] else -1 for y in self.Y])
        weights     = uniform_vector(self.n_kernels)
        ker_matrix  = self.func_form(self.KL, weights)
        alpha,r2    = opt_radius(ker_matrix)
        gamma,m2    = opt_margin(ker_matrix, YY)
        obj         = (r2 / m2) / len(self.Y)

        #caching
        self.cache.YY = YY
        self.cache.alpha = alpha
        self.cache.gamma = gamma

        return Solution(
            weights=weights, 
            objective=obj,
            ker_matrix=ker_matrix,
            )

        


    def do_step(self):

        YY    = self.cache.YY
        alpha = self.cache.alpha
        gamma = self.cache.gamma

        beta = np.log(self.solution.weights)
        beta = self._update_grad(self.solution.ker_matrix, YY, beta, alpha, gamma)

        w = np.exp(beta)
        w /= sum(w)
        ker_matrix = self.func_form(self.KL, w)
        try :
            # try warm start in radius/margin reoptimization
            new_alpha,r2 = opt_radius(ker_matrix   ,init_sol=alpha)
            new_gamma,m2 = opt_margin(ker_matrix,YY,init_sol=gamma)
        except :
            new_alpha,r2 = opt_radius(ker_matrix   )
            new_gamma,m2 = opt_margin(ker_matrix,YY)
        obj = (r2 / m2) / len(self.Y)

        self.cache.alpha = new_alpha
        self.cache.gamma = new_gamma

        ker_matrix = self.func_form(self.KL, w)
        return Solution(
            weights=w,
            objective=obj,
            ker_matrix=ker_matrix
            )

        

    def _update_grad(self,Kc, YY, beta, alpha, gamma):
        n = self.n_kernels
        
        eb = np.exp(beta)
        a,b = [], []
        gammaY = gamma['x'].T*YY
        for K in self.KL:   #optimized for generators
            K = matrix(K)
            a.append( 1.-(alpha['x'].T*matrix(K)*alpha['x'])[0] )
            b.append( (gammaY*matrix(K)*gammaY.T)[0] )
        ebb, eba = np.dot(eb,b), np.dot(eb,a)
        den = np.dot(eb,b)**2
        num = [eb[r] * (a[r]*ebb - b[r]*eba) for r in range(n)]
        
        new_beta = np.array([beta[k] - self.learning_rate * (num[k]/den) for k in range(n)])
        return new_beta

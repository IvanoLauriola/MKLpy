# -*- coding: latin-1 -*-
"""
"""

from .AlternateMKL import AlternateMKL
from ..arrange import average, summation
from ..lists.generator import HPK_generator
from sklearn.svm import SVC 
from cvxopt import matrix, spdiag, solvers
import numpy as np
import time,sys 



def radius(K,lam=0,init_sol=None): 
    n = K.shape[0]
    K = matrix(K)
    P = 2 * ( (1-lam) * K + spdiag([lam]*n) )
    p = -matrix([K[i,i] for i in range(n)])
    G = -spdiag([1.0] * n)
    h = matrix([0.0] * n)
    A = matrix([1.0] * n).T
    b = matrix([1.0])
    solvers.options['show_progress']=False
    sol = solvers.qp(P,p,G,h,A,b,initvals=init_sol)
    radius2 = (-p.T * sol['x'])[0] - (sol['x'].T * K * sol['x'])[0]
    return sol, radius2


def margin(K,Y,lam=0,init_sol=None):
    n = len(Y)
    YY = spdiag(list(Y))
    K = matrix(K)
    lambdaDiag = spdiag([lam]*n)
    P = 2*( (1-lam) * (YY*K*YY) + lambdaDiag )
    p = matrix([0.0]*n)
    G = -spdiag([1.0]*n)
    h = matrix([0.0]*n)
    A = matrix([[1.0 if Y[i]==+1 else 0 for i in range(n)],
                [1.0 if Y[j]==-1 else 0 for j in range(n)]]).T
    b = matrix([[1.0],[1.0]],(2,1))
    solvers.options['show_progress']=False
    sol = solvers.qp(P,p,G,h,A,b,initvals=init_sol)
    margin2 = sol['dual objective'] - (sol['x'].T * lambdaDiag * sol['x'])[0]
    return sol,margin2

def opt_margin(K,YY,init_sol=None):
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

    def __init__(self, learner=SVC(C=1000), generator=HPK_generator(n=10), multiclass_strategy='ova', verbose=False,
                max_iter=1000, learning_rate=0.01, callbacks=[]):
        super().__init__(learner=learner, generator=generator, multiclass_strategy=multiclass_strategy, max_iter=max_iter, verbose=verbose, callbacks=callbacks)
        self.func_form = summation


    def get_params(self, deep=True):
        return super().get_params()


    def initialize_optimization(self):
        context = {'YY' : spdiag([1 if y==self.Y[0] else -1 for y in self.Y])}
        weights = np.ones(self.n_kernels)/self.n_kernels
        ker_matrix = self.func_form(self.KL, weights)

        alpha,r2 = radius(ker_matrix)
        gamma,m2 = opt_margin(ker_matrix, context['YY'])
        incumbent_solution = { 'gamma': gamma,
                               'alpha': alpha}
        obj = (r2 / m2) / len(self.Y)

        return obj, incumbent_solution, weights, ker_matrix, context

        


    def do_step(self):

        YY = self.context['YY']

        alpha = self.incumbent_solution['alpha']
        gamma = self.incumbent_solution['gamma']

        beta = np.log(self.weights)
        beta = self._update_grad(self.ker_matrix, YY, beta, alpha, gamma)

        w = np.exp(beta)
        w /= sum(w)
        ker_matrix = self.func_form(self.KL, w)
        try :
            new_alpha,r2 = radius(ker_matrix       ,init_sol=context['alpha'].copy())
            new_gamma,m2 = opt_margin(ker_matrix,YY,init_sol=context['gamma'].copy())
        except :
            new_alpha,r2 = radius(ker_matrix   )
            new_gamma,m2 = opt_margin(ker_matrix,YY)
        obj = (r2 / m2) / len(self.Y)
        incumbent_solution = {'alpha': new_alpha, 'gamma': new_gamma}

        return obj, incumbent_solution, w, ker_matrix

        

    def _update_grad(self,Kc, YY, beta, alpha, gamma):
        
        eb = np.exp(np.array(beta))

        a = np.array([1.0-(alpha['x'].T*matrix(K)*alpha['x'])[0] for K in self.KL])
        b = np.array([(gamma['x'].T*YY*matrix(K)*YY*gamma['x'])[0] for K in self.KL])            
        den = [np.dot(eb,b)**2]*self.n_kernels
        num = [eb[r] * (a[r]*np.dot(eb,b)   -   b[r]*np.dot(eb,a)) for r in range(self.n_kernels)]
        
        new_beta = np.array([beta[k] - self.learning_rate * (num[k]/den[k]) for k in range(self.n_kernels)])
        new_beta = new_beta - new_beta.max()
        return new_beta

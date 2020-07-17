# -*- coding: latin-1 -*-

"""
@author: Ivano Lauriola
@email: ivano.lauriola@phd.unipd.it, ivanolauriola@gmail.com

This file is part of MKLpy: a scikit-compliant framework for Multiple Kernel Learning
This file is distributed with the GNU General Public License v3 <http://www.gnu.org/licenses/>.  

"""


from . import Solution, Cache, TwoStepMKL
from ..arrange import average, summation
from ..utils.misc import uniform_vector
from sklearn.svm import SVC 
from cvxopt import matrix, spdiag, solvers
import numpy as np
import time,sys 
import torch
from ..metrics import ratio



def opt_radius(K, init_sol=None): 
    n = K.shape[0]
    K = matrix(K.numpy())
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
    P = 2 * (YY * matrix(K.numpy()) * YY)
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





class GRAM(TwoStepMKL):

    direction = 'min'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.func_form = summation


    def get_params(self, deep=True):
        return super().get_params()


    def initialize_optimization(self):
        YY          = spdiag([1 if y==self.classes_[1] else -1 for y in self.Y])
        Y           = torch.tensor([1 if y==self.classes_[1] else -1 for y in self.Y])
        weights     = uniform_vector(self.n_kernels)
        ker_matrix  = self.func_form(self.KL, weights)
        alpha,r2    = opt_radius(ker_matrix)
        gamma,m2    = opt_margin(ker_matrix, YY)
        obj         = (r2 / m2) / len(self.Y)

        #caching
        self.cache.YY = YY
        self.cache.Y  = Y
        self.cache.alpha = alpha
        self.cache.gamma = gamma

        dual_coef    = torch.Tensor(np.array(gamma['x'])).double().T[0]
        bias = 0.5 * (dual_coef @ ker_matrix @ (dual_coef.T * Y)).item()

        return Solution(
            weights=weights, 
            objective=obj,
            ker_matrix=ker_matrix,
            dual_coef = dual_coef,
            bias = bias,
            )

        


    def do_step(self):

        YY    = self.cache.YY
        alpha = self.cache.alpha
        gamma = self.cache.gamma

        beta = np.log(self.solution.weights)
        beta = self._update_grad(self.solution.ker_matrix, YY, beta, alpha, gamma)

        w = np.exp(beta)
        w /= w.sum()
        ker_matrix = self.func_form(self.KL, w)
        try :
            # try incremental radius/margin optimization
            new_alpha,r2 = opt_radius(ker_matrix   ,init_sol=alpha)
            new_gamma,m2 = opt_margin(ker_matrix,YY,init_sol=gamma)
        except :
            new_alpha,r2 = opt_radius(ker_matrix   )
            new_gamma,m2 = opt_margin(ker_matrix,YY)
        obj = (r2 / m2) / len(self.Y)

        self.cache.alpha = new_alpha
        self.cache.gamma = new_gamma

        ker_matrix = self.func_form(self.KL, w)

        dual_coef    = torch.Tensor(np.array(new_gamma['x'])).double().T[0]
        yg = dual_coef.T * self.cache.Y
        bias = 0.5 * (dual_coef @ ker_matrix @ yg).item()
        return Solution(
            weights=w,
            objective=obj,
            ker_matrix=ker_matrix,
            dual_coef = dual_coef,
            bias = bias,
            )

        

    def _update_grad(self,Kc, YY, beta, alpha, gamma):
        n = self.n_kernels
        gammaY = gamma['x'].T * YY
        gammaY = torch.DoubleTensor(np.array(gammaY))
        gamma = torch.DoubleTensor(np.array(gamma['x']))
        alpha = torch.DoubleTensor(np.array(alpha['x']))
        
        eb = torch.exp(torch.tensor(beta))
        a,b = [], []
        for K in self.KL:   #optimized for generators
            #K = matrix(K.numpy().astype(np.double))
            #a.append( 1.-(alpha['x'].T*matrix(K)*alpha['x'])[0] )
            #b.append( (gammaY*matrix(K)*gammaY.T)[0] )
            a.append(1. - (alpha.T @ K @ alpha).item())
            b.append( (gammaY @ K @ gammaY.T).item() )
        a, b = torch.tensor(a,  dtype=torch.double), torch.tensor(b,  dtype=torch.double)
        ebb, eba = (eb @ b).item(), (eb @ a).item()
        den = ((eb @ b)**2).item()
        num = [eb[r] * (a[r]*ebb - b[r]*eba) for r in range(n)]
        
        new_beta = np.array([beta[k] - self.learning_rate * (num[k]/den) for k in range(n)])
        return new_beta

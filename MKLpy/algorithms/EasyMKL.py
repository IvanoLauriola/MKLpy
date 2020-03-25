# -*- coding: latin-1 -*-
"""
@author: Ivano Lauriola and Michele Donini
@email: ivano.lauriola@phd.unipd.it
 
EasyMKL: a scalable multiple kernel learning algorithm
by Fabio Aiolli and Michele Donini

This file is part of MKLpy: a scikit-compliant framework for Multiple Kernel Learning
This file is distributed with the GNU General Public License v3 <http://www.gnu.org/licenses/>. 
 
Paper @ http://www.math.unipd.it/~mdonini/publications.html
"""

from .base import MKL, Solution
from .komd import KOMD
from ..multiclass import OneVsOneMKLClassifier as ovoMKL, OneVsRestMKLClassifier as ovaMKL
from ..arrange import summation
from ..utils.exceptions import BinaryProblemError

from cvxopt import matrix, spdiag, solvers
import numpy as np
 

 
 
class EasyMKL(MKL):
    ''' EasyMKL is a Multiple Kernel Learning algorithm.
        The parameter lam (lambda) has to be validated from 0 to 1.
 
        For more information:
        EasyMKL: a scalable multiple kernel learning algorithm
            by Fabio Aiolli and Michele Donini
 
        Paper @ http://www.math.unipd.it/~mdonini/publications.html
    '''
    def __init__(self, learner=KOMD(lam=0.1), lam=0.1, multiclass_strategy='ova', verbose=False):
        super().__init__(learner=learner, multiclass_strategy=multiclass_strategy, verbose=verbose)
        self.func_form = summation
        self.lam = lam


        
    def _combine_kernels(self):
        assert len(np.unique(self.Y)) == 2
        Y = [1 if y==self.classes_[1] else -1 for y in self.Y]
        n_sample = len(self.Y)
        ker_matrix = matrix(self.func_form(self.KL))
        YY = spdiag(Y)
        #KLL = (1.0-self.lam)*YY*ker_matrix*YY
        #LID = spdiag([self.lam]*n_sample)
        #Q = 2*(KLL+LID)
        Q = 2 * ((1.0-self.lam)*YY*ker_matrix*YY + spdiag([self.lam]*n_sample))
        p = matrix([0.0]*n_sample)
        G = -spdiag([1.0]*n_sample)
        h = matrix([0.0]*n_sample,(n_sample,1))
        A = matrix([[1.0 if lab==+1 else 0 for lab in Y],[1.0 if lab2==-1 else 0 for lab2 in Y]]).T
        b = matrix([[1.0],[1.0]],(2,1))
         
        solvers.options['show_progress'] = False
        solvers.options['maxiters'] = 200
        sol = solvers.qp(Q,p,G,h,A,b)
        gamma = sol['x']

        yg = gamma.T * YY
        weights = [(yg*matrix(K)*yg.T)[0] for K in self.KL]
         
        norm2 = sum([w for w in weights])
        
        weights = np.array([w / norm2 for w in weights])
        ker_matrix = self.func_form(self.KL, weights)
        return Solution(
            weights=weights,
            objective=None,
            ker_matrix=ker_matrix,
            )

 
    def get_params(self, deep=True):
        # this estimator has parameters:
        new_params = {'lam': self.lam}
        return super().get_params().update(new_params)

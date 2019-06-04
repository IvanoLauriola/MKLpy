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

from .base import MKL
from .komd import KOMD
from ..multiclass import OneVsOneMKLClassifier as ovoMKL, OneVsRestMKLClassifier as ovaMKL
from ..arrange import summation
from ..utils.exceptions import BinaryProblemError
from ..lists import HPK_generator

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
    def __init__(self, learner=KOMD(lam=0.1), lam=0.1, generator=HPK_generator(n=10), multiclass_strategy='ova', verbose=False):
        super(self.__class__, self).__init__(learner=learner, generator=generator, multiclass_strategy=multiclass_strategy, func_form=summation, verbose=verbose)
        self.lam = lam

        
    def _combine_kernels(self):
        Y = [1 if y==self.classes_[1] else -1 for y in self.Y]
        n_sample = len(self.Y)
        ker_matrix = matrix(summation(self.KL))
        YY = spdiag(Y)
        KLL = (1.0-self.lam)*YY*ker_matrix*YY
        LID = spdiag([self.lam]*n_sample)
        Q = 2*(KLL+LID)
        p = matrix([0.0]*n_sample)
        G = -spdiag([1.0]*n_sample)
        h = matrix([0.0]*n_sample,(n_sample,1))
        A = matrix([[1.0 if lab==+1 else 0 for lab in Y],[1.0 if lab2==-1 else 0 for lab2 in Y]]).T
        b = matrix([[1.0],[1.0]],(2,1))
         
        solvers.options['show_progress'] = False
        solvers.options['maxiters'] = 200
        sol = solvers.qp(Q,p,G,h,A,b)
        gamma = sol['x']
        if self.verbose:
            print ('[EasyMKL]')
            print ('optimization finished, #iter = %d' % sol['iterations'])
            print ('status of the solution: %s' % sol['status'])
            print ('objval: %.5f' % sol['primal objective'])

        yg = gamma.T * YY
        weights = [(yg*matrix(K)*yg.T)[0] for K in self.KL]
         
        norm2 = sum([w for w in weights])
        self.weights = np.array([w / norm2 for w in weights])
        ker_matrix = summation(self.KL, self.weights)
        self.ker_matrix = ker_matrix
        return ker_matrix

 
    def get_params(self, deep=True):
        # this estimator has parameters:
        return {"lam": self.lam,
                "generator": self.generator, "max_iter":self.max_iter,
                "verbose":self.verbose, "multiclass_strategy":self.multiclass_strategy,
                'learner':self.learner}

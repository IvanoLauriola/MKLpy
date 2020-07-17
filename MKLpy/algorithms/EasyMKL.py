# -*- coding: latin-1 -*-

"""
@author: Ivano Lauriola
@email: ivano.lauriola@phd.unipd.it, ivanolauriola@gmail.com

This file is part of MKLpy: a scikit-compliant framework for Multiple Kernel Learning
This file is distributed with the GNU General Public License v3 <http://www.gnu.org/licenses/>.  

"""

from .base import OneStepMKL, Solution
from .komd import KOMD
from ..multiclass import OneVsOneMKLClassifier as ovoMKL, OneVsRestMKLClassifier as ovaMKL
from ..arrange import summation
from ..utils.exceptions import BinaryProblemError
from ..utils.misc import identity_kernel
from ..metrics import margin

import torch
import numpy as np

 
 
class EasyMKL(OneStepMKL):
    ''' EasyMKL is a Multiple Kernel Learning algorithm.
        The parameter lam (lambda) has to be validated from 0 to 1.
 
        For more information:
        EasyMKL: a scalable multiple kernel learning algorithm
            by Fabio Aiolli and Michele Donini
 
        Paper @ http://www.math.unipd.it/~mdonini/publications.html
    '''

    

    def __init__(self, learner=KOMD(lam=0.1), lam=0.1, solver='auto', **kwargs):
        super().__init__(learner=learner, **kwargs)
        self.lam    = lam
        self.solver = solver
        if not (0 <= lam <= 1):
            raise ValueError('The lam value has to be in [0, 1]')
        if solver not in ['auto', 'libsvm', 'cvxopt']:
            raise ValueError('The solver has to be in [\'auto\', \'libsvm\', \'cvxopt\']')

        if solver == 'auto':
            self._solver = 'libsvm' if self.lam > 0.01 else 'cvxopt'
        else:
            self._solver = solver
        self.func_form = summation


        
    def _combine_kernels(self):

        assert len(self.Y.unique()) == 2
        Y = torch.tensor([1 if y==self.classes_[1] else -1 for y in self.Y])
        n_sample = len(self.Y)

        self.func_form(self.KL)

        ker_matrix = (1-self.lam) * self.func_form(self.KL) + self.lam * identity_kernel(n_sample)

        mar, gamma = margin(
            ker_matrix, Y, 
            return_coefs    = True, 
            solver          = self._solver, 
            max_iter        = self.max_iter, 
            tol             = self.tolerance)
        yg = gamma.T * Y
        weights = torch.tensor([(yg.view(n_sample, 1).T @ K @ yg).item() for K in self.KL])
        weights = weights / weights.sum()

        ker_matrix = self.func_form(self.KL, weights)
        bias = 0.5 * (gamma @ ker_matrix @ yg).item()
        return Solution(
            weights     = weights,
            objective   = None,
            ker_matrix  = ker_matrix,
            dual_coef   = None,
            bias        = None,
            )

 
    def get_params(self, deep=True):
        # this estimator has parameters:
        params = super().get_params()
        params.update({'lam': self.lam})
        return params

    def score(self, KL):
        raise Error('EasyMKL does not support the score function. Use a scikit-compliant base learner')

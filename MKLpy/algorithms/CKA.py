# -*- coding: latin-1 -*-

"""
@author: Ivano Lauriola
@email: ivano.lauriola@phd.unipd.it, ivanolauriola@gmail.com

This file is part of MKLpy: a scikit-compliant framework for Multiple Kernel Learning
This file is distributed with the GNU General Public License v3 <http://www.gnu.org/licenses/>.  

"""

from .base import OneStepMKL, Solution
from ..arrange import summation
from ..metrics import margin, alignment_yy, alignment
from ..preprocessing import kernel_centering
from ..utils.misc import ideal_kernel
import torch
from sklearn.svm import SVC 

 
 
class CKA(OneStepMKL):
    ''' 
        Centered Kernel Alignment
        Paper:
            Cortes, C., Mohri, M., & Rostamizadeh, A. (2010). Two-stage learning kernel algorithms.
    '''

    def __init__(self, learner=SVC(C=1000), **kwargs):
        super().__init__(learner=learner, **kwargs)
        self.func_form = summation
        
    def _combine_kernels(self):
        yy = ideal_kernel(self.Y).flatten()
        a = torch.tensor([kernel_centering(K).flatten() @ yy for K in self.KL])

        #Warning: kernels are centered multiple times to save memory!
        M = torch.zeros(self.n_kernels, self.n_kernels, dtype=torch.double)
        for i in range(self.n_kernels):
            Ki = kernel_centering(self.KL[i]).flatten()
            M[i,i] = Ki @ Ki
            for j in range(i):
                Kj = kernel_centering(self.KL[j]).flatten()
                M[i,j] = M[j,i] = Ki @ Kj
        Minv = M.inverse()
        
        weights = Minv @ a
        weights /= weights.norm(p=2)
        ker_matrix = self.func_form(self.KL, weights)
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
        return params

    def score(self, KL):
        raise Error('HeuristicMKL does not support the score function. Use a scikit-compliant base learner')


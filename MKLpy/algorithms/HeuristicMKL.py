# -*- coding: latin-1 -*-

"""
@author: Ivano Lauriola
@email: ivano.lauriola@phd.unipd.it, ivanolauriola@gmail.com

This file is part of MKLpy: a scikit-compliant framework for Multiple Kernel Learning
This file is distributed with the GNU General Public License v3 <http://www.gnu.org/licenses/>.  

"""

from .base import OneStepMKL, Solution
from ..arrange import summation
from ..metrics import margin, alignment_yy
import torch
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

 
 
class HeuristicMKL(OneStepMKL):
    ''' 
    '''

    def __init__(self, learner=SVC(C=1000),**kwargs):
        super().__init__(learner=learner, **kwargs)
        self.func_form = summation
        
    def _combine_kernels(self):
        weights = torch.tensor([self._eval(K, self.Y) for K in self.KL])
        weights = self._transform(weights)
        weights = weights / weights.sum()
        ker_matrix = self.func_form(self.KL, weights)
        return Solution(
            weights     = weights,
            objective   = None,
            ker_matrix  = ker_matrix,
            dual_coef   = None,
            bias        = None,
            )

    def _eval(self, K, Y):
        raise NotImplementedError('This method has to be implemented in the derived class')

    def _transform(self, weights):
    	return weights

    def get_params(self, deep=True):
        # this estimator has parameters:
        return super().get_params()

    def score(self, KL):
        raise Error('HeuristicMKL does not support the score function. Use a scikit-compliant base learner')





class PWMK(HeuristicMKL):
    ''' 
    	paper: 
    		Tanabe, H., Ho, T. B., Nguyen, C. H., & Kawasaki, S. (2008, July). 
    		Simple but effective methods for combining kernels in computational biology. 
    		In 2008 IEEE International Conference on Research, Innovation and Vision for the Future in Computing 
    			and Communication Technologies (pp. 71-78). IEEE.
    '''

    def __init__(self, delta=0, cv=3, **kwargs):
        super().__init__(**kwargs)
        if not (0 <= delta <= 1):
            raise ValueError ('sigma has to be between 0 and 1')
        self.delta = delta
        self.cv = cv

    def _eval(self, K, Y):
    	return cross_val_score(self.learner, K.cpu().numpy(), Y.cpu().numpy(), scoring='accuracy', cv=self.cv).mean()

    def _transform(self, weights):
        m = weights.min()
        abs_delta = m * self.delta
        weights = weights - abs_delta
        assert weights.min().item() >= 0.0
        return weights

    def get_params(self, deep=True):
        # this estimator has parameters:
        params = super().get_params()
        params.update({'delta': self.delta, 'cv': self.cv})
        return params


class FHeuristic(HeuristicMKL):
	'''
		paper:
		Qiu, S., & Lane, T. (2008). 
		A framework for multiple kernel support vector regression and its applications to siRNA efficacy prediction. 
		IEEE/ACM Transactions on Computational Biology and Bioinformatics, 6(2), 190-199.
	'''

	def __init__(self, **kwargs):
		super().__init__(**kwargs)

	def _eval(self, K, Y):
		return alignment_yy(K, Y)
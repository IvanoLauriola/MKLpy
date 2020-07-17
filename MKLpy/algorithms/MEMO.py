# -*- coding: latin-1 -*-

"""
@author: Ivano Lauriola
@email: ivano.lauriola@phd.unipd.it, ivanolauriola@gmail.com

This file is part of MKLpy: a scikit-compliant framework for Multiple Kernel Learning
This file is distributed with the GNU General Public License v3 <http://www.gnu.org/licenses/>.  

"""

from . import Solution, Cache, TwoStepMKL
from ..arrange import average, summation
from ..metrics import frobenius, margin
import numpy as np
from cvxopt import spdiag,matrix,solvers
import torch
from ..utils.misc import uniform_vector


class MEMO(TwoStepMKL):

	direction = 'max'

	def __init__(self, theta=0.0, min_margin=1e-4, solver='auto', **kwargs):
		super().__init__(**kwargs)
		self.func_form = summation
		self.theta = theta
		self.min_margin = min_margin
		
		self._solver = 'libsvm' if solver in ['auto', 'libsvm'] else 'cvxopt'


	def get_params(self, deep=True):
		new_params = {'theta': self.theta,
				'min_margin':self.min_margin}
		params = super().get_params()
		params.update(new_params)
		return params


	def initialize_optimization(self):
		Q = torch.tensor([[self.KL[j].flatten() @ self.KL[i].flatten() for j in range(self.n_kernels)] for i in range(self.n_kernels)])
		print (Q.size())
		Q /= (torch.diag(Q).sum() / self.n_kernels)

		self.cache.Q  = Q
		self.cache.Y  = torch.tensor([1 if y==self.classes_[1] else -1 for y in self.Y])

		weights    = uniform_vector(self.n_kernels)
		ker_matrix = self.func_form(self.KL, weights)
		mar, gamma = margin(
			ker_matrix, self.cache.Y, 
			return_coefs    = True, 
			solver          = self._solver, 
			#max_iter        = self.max_iter*10, 
			tol             = self.tolerance)
		yg = gamma.T * self.cache.Y

		self.cache.gamma  = gamma
		self.cache.margin = mar
		bias = 0.5 * (gamma @ ker_matrix @ yg).item()

		print (weights.size(), yg.size(), weights.T.size(), yg.T.size(), ker_matrix.size(), Q.size(), len(self.KL))

		obj = (yg.T @ ker_matrix @ yg).item() + (weights @ Q @ weights).item() * self.theta *.5
		print ('ok')
		return Solution(
			weights     = weights,
			objective   = obj,
			ker_matrix  = ker_matrix,
			bias 		= bias,
			dual_coef   = gamma
		)


	def do_step(self):
		Y  = self.cache.Y

		# positive margin constraint
		if self.cache.margin <= self.min_margin: # prevents initial negative margin. Looking for a better solution
			return self.solution

		# weights update
		yg = self.cache.gamma.T * Y
		grad = torch.tensor([self.theta * (qv @ self.solution.weights).item() + (yg.T  @ K @ yg).item()  \
				for qv,K in zip(self.cache.Q, self.KL)])
		beta = self.solution.weights#.log()
		beta = beta + self.learning_rate * grad
		beta_e = beta#.exp()

		weights = beta_e
		weights[weights<0] = 0
		weights /= sum(beta_e)

		# compute combined kernel
		ker_matrix = self.func_form(self.KL, weights)

		# margin (and gamma) update
		mar, gamma = margin(
			ker_matrix, Y, 
			return_coefs    = True, 
			solver          = self._solver, 
			#max_iter        = self.max_iter, 
			tol             = self.tolerance)

		# positive margin constraint
		if mar <= self.min_margin:
			return self.solution

		# compute objective and bias
		yg = gamma.T * Y

		obj = (yg.T @ ker_matrix @ yg).item() + self.theta *.5 * (weights.view(1,len(weights)) @ self.cache.Q @ weights).item()
		bias = 0.5 * (gamma @ ker_matrix @ yg).item()

		#update cache
		self.cache.gamma  = gamma
		self.cache.margin = mar
		
		return Solution(
			weights    = weights,
			objective  = obj,
			ker_matrix = ker_matrix,
            dual_coef = gamma,
            bias = bias,
		)

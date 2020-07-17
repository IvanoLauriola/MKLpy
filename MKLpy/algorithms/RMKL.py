# -*- coding: latin-1 -*-

"""
@author: Ivano Lauriola
@email: ivano.lauriola@phd.unipd.it, ivanolauriola@gmail.com

This file is part of MKLpy: a scikit-compliant framework for Multiple Kernel Learning
This file is distributed with the GNU General Public License v3 <http://www.gnu.org/licenses/>.  

"""

from . import Solution, Cache, TwoStepMKL
from ..arrange import average, summation
from ..metrics import frobenius, radius
import numpy as np
from cvxopt import spdiag,matrix,solvers
import torch
from ..utils.misc import uniform_vector
from sklearn.svm import SVC


class RMKL(TwoStepMKL):

	direction = 'max'

	def __init__(self, C=1.0, solver='auto', **kwargs):
		super().__init__(**kwargs)
		self.func_form = summation
		self.C = C
		
		self._solver = 'libsvm' if solver in ['auto', 'libsvm'] else 'cvxopt'


	def get_params(self, deep=True):
		new_params = {'C': self.C}
		params = super().get_params()
		params.update(new_params)
		return params


	def initialize_optimization(self):
		self.cache.R = torch.DoubleTensor([radius(K)**2 for K in self.KL])
		self.cache.Y = torch.tensor([1 if y==self.classes_[1] else -1 for y in self.Y])

		weights    = uniform_vector(self.n_kernels)
		ker_matrix = self.func_form(self.KL, weights)

		mar, gamma = self._get_gamma(ker_matrix, self.cache.Y)
		yg = gamma.T * self.cache.Y
		bias = 0.5 * (gamma @ ker_matrix @ yg).item()

		coef_r = (self.cache.R.T @ weights).item() / self.C
		obj = (yg.T @ (ker_matrix + torch.ones(len(gamma)).diag()*coef_r ) @ yg).item() + gamma.sum() + 0

		return Solution(
			weights     = weights,
			objective   = obj,
			ker_matrix  = ker_matrix,
			bias 		= bias,
			dual_coef   = gamma
		)


	def do_step(self):
		Y  = self.cache.Y

		# weights update
		coef_r = (self.cache.R.T @ self.solution.weights).item() / self.C
		yg = self.solution.dual_coef.T * Y

		grad = torch.tensor([(yg.T  @ (K + rk/self.C * torch.ones(len(yg)).diag()) @ yg).item() \
				for rk,K in zip(self.cache.R, self.KL)])
		weights = self.solution.weights + self.learning_rate * grad
		weights[weights<0] = 0
		weights /= sum(weights)

		# compute combined kernel
		ker_matrix = self.func_form(self.KL, weights)

		# margin (and gamma) update
		mar, gamma = self._get_gamma(ker_matrix, self.cache.Y)

		# compute objective and bias
		yg = gamma.T * Y
		coef_r = (self.cache.R.T @ weights).item() / self.C
		obj = (yg.T @ (ker_matrix + torch.ones(len(gamma)).diag()*coef_r ) @ yg).item() + gamma.sum() + 0
		bias = 0.5 * (gamma @ ker_matrix @ yg).item()
		
		return Solution(
			weights    = weights,
			objective  = obj,
			ker_matrix = ker_matrix,
			dual_coef = gamma,
			bias = bias,
		)


	def _get_gamma(self, K, Y):
		svm = SVC(C=self.C, kernel='precomputed').fit(K,Y)
		n = len(Y)
		gamma = torch.zeros(n).double()
		gamma[svm.support_] = torch.tensor(svm.dual_coef_)
		idx_pos = gamma > 0
		idx_neg = gamma < 0
		sum_pos, sum_neg = gamma[idx_pos].sum(), gamma[idx_neg].sum()
		gamma[idx_pos] /= sum_pos
		gamma[idx_neg] /= sum_neg
		gammay = gamma * Y
		obj = (gammay.view(n,1).T @ K @ gammay).item() **.5
		return obj, gamma

# -*- coding: latin-1 -*-

"""
@author: Ivano Lauriola
@email: ivano.lauriola@phd.unipd.it

This file is part of MKLpy: a scikit-compliant framework for Multiple Kernel Learning
This file is distributed with the GNU General Public License v3 <http://www.gnu.org/licenses/>.  

"""

from .base import MKL, Solution
from ..arrange import average
from sklearn.svm import SVC
import torch


class AverageMKL(MKL):

	def __init__(self, 
		learner=SVC(C=1000), 
		multiclass_strategy='ova', 
		verbose=False,
		max_iter=-1,
		tolerance=1e-7,
		solver='auto'
		):
		super().__init__(
			learner=learner, 
			multiclass_strategy=multiclass_strategy, 
			verbose=verbose,
			max_iter=max_iter,
			tolerance=tolerance,
			solver=solver,
		)
		
		#set other params
		self.func_form = average


	def _combine_kernels(self):
		n = self.n_kernels
		w = torch.ones(n)/n
		return Solution(
			weights		= w,
			objective	= None,
			ker_matrix	= self.func_form(self.KL, w),
			)


	def get_params(self, deep=True):
		# no further parameters are introduced in AverageMKL
		return super().get_params()

# -*- coding: latin-1 -*-

"""
@author: Ivano Lauriola
@email: ivano.lauriola@phd.unipd.it, ivanolauriola@gmail.com

This file is part of MKLpy: a scikit-compliant framework for Multiple Kernel Learning
This file is distributed with the GNU General Public License v3 <http://www.gnu.org/licenses/>.  

"""

from .base import OneStepMKL, Solution
from ..arrange import average
from sklearn.svm import SVC
import torch


class AverageMKL(OneStepMKL):

	def __init__(self, learner=SVC(C=1000), **kwargs):
		super().__init__(learner=learner, **kwargs)
		self.func_form = average


	def _combine_kernels(self):
		n = self.n_kernels
		w = torch.ones(n)/n
		return Solution(
			weights		= w,
			objective	= None,
			ker_matrix	= self.func_form(self.KL, w),
			dual_coef = None,
			bias = None
			)


	def get_params(self, deep=True):
		# no further parameters are introduced in AverageMKL
		return super().get_params()

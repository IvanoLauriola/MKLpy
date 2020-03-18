# -*- coding: latin-1 -*-

"""
@author: Ivano Lauriola
@email: ivano.lauriola@phd.unipd.it

This file is part of MKLpy: a scikit-compliant framework for Multiple Kernel Learning
This file is distributed with the GNU General Public License v3 <http://www.gnu.org/licenses/>.  

"""

from .base import MKL
from ..arrange import average
from ..utils.misc import uniform_vector
from sklearn.svm import SVC
import numpy as np


class AverageMKL(MKL):

	def __init__(self, learner=SVC(C=1000), multiclass_strategy='ova', verbose=False):
		super().__init__(learner=learner, multiclass_strategy=multiclass_strategy, verbose=verbose)
		#set other params
		self.func_form = average


	def _combine_kernels(self):
		self.weights = uniform_vector(self.n_kernels)	# weights vector is a np.ndarray
		self.ker_matrix = self.func_form(self.KL,self.weights)			# combine kernels
		return self.ker_matrix

	def get_params(self, deep=True):
		# no further parameters are introduced in AverageMKL
		return super().get_params()

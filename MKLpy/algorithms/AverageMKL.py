# -*- coding: latin-1 -*-

"""
@author: Ivano Lauriola
@email: ivano.lauriola@phd.unipd.it

This file is part of MKLpy: a scikit-compliant framework for Multiple Kernel Learning
This file is distributed with the GNU General Public License v3 <http://www.gnu.org/licenses/>.  

"""

from .base import MKL
from ..arrange import average
from ..lists.generator import HPK_generator
from sklearn.svm import SVC
import numpy as np


class AverageMKL(MKL):

	def __init__(self, learner=SVC(C=1000), generator=HPK_generator(n=10), multiclass_strategy='ova', verbose=False):
		super(self.__class__, self).__init__(learner=learner, generator=generator, multiclass_strategy=multiclass_strategy, func_form=average, verbose=verbose)
		#set other params


	def _combine_kernels(self):
		self.weights = np.ones(self.n_kernels)/self.n_kernels	# weights vector is a np.ndarray
		self.ker_matrix = self.func_form(self.KL,self.weights)			# combine kernels
		return self.ker_matrix

	def get_params(self, deep=True):
		return {"learner":self.learner,
				"generator":self.generator,
				"verbose":self.verbose,
				"multiclass_strategy":self.multiclass_strategy}

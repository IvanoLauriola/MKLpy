# -*- coding: latin-1 -*-

"""
@author: Ivano Lauriola
@email: ivano.lauriola@phd.unipd.it

This file is part of MKLpy: a scikit-compliant framework for Multiple Kernel Learning
This file is distributed with the GNU General Public License v3 <http://www.gnu.org/licenses/>.  

"""

from .base import MKL
from ..arrange import summation
from ..lists.generator import HPK_generator
from sklearn.svm import SVC
import numpy as np



class AlternateMKL(MKL):


	def __init__(self, learner=SVC(C=1000), generator=HPK_generator(n=10), multiclass_strategy='ova', verbose=False,
				 max_iter=1000, learning_rate=0.01, callbacks=[]):
		super().__init__(learner=learner, generator=generator, multiclass_strategy=multiclass_strategy, verbose=verbose)
		self.max_iter 		= max_iter
		self.learning_rate	= learning_rate
		self.callbacks		= callbacks


	def get_params(self, deep=True):
		# this estimator has parameters:
		new_params = {'max_iter' : self.max_iter,
					  'learning_rate': self.learning_rate,
					  'callbacks': self.callbacks}
		params = super().get_params()
		params.update(new_params)
		return params



	@classmethod
	def initialize_optimization(self):
		''' this method initialize the initial context, data structures for the optimization,
			and the initial weights vector'''
		raise NotImplementedError('This method has to be implemented in the derived class')
		

	@classmethod
	def do_step(self, sol):
		'''this method computes an optimization step. Sol is the incumbent solution'''
		raise NotImplementedError('This method has to be implemented in the derived class')


	def _combine_kernels(self):

		# initialize optimization problem and weights
		initial_result = self.initialize_optimization()
		self.obj, self.incumbent_solution, self.weights, self.ker_matrix, self.context = initial_result

		self.is_fitted = True

		# initialize callbacks
		for callback in self.callbacks:
			callback.on_train_begin(self)

		##########################
		# MAIN OPTIMIZATION LOOP #
		##########################
		for step in range(1, self.max_iter + 1):

			# callbacks on_step_begin
			for callback in self.callbacks:
				callback.on_step_begin(step)

			_obj, _sol, _weights, _ker_matrix = self.do_step()
			if _obj == self.obj:
				break

			self.obj                = _obj
			self.incumbent_solution = _sol
			self.weights            = _weights
			self.ker_matrix         = _ker_matrix


			# callbacks on_step_end
			for callback in self.callbacks:
				callback.on_step_end(step)
		##########################
		# STOP OPTIMIZATION LOOP #
		##########################

		self.ker_matrix = self.func_form(self.KL,self.weights)

		# callbacks optimization end
		for callback in self.callbacks:
			callback.on_train_end()

		return self.ker_matrix





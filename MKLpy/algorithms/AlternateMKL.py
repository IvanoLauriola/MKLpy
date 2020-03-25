# -*- coding: latin-1 -*-

"""
@author: Ivano Lauriola
@email: ivano.lauriola@phd.unipd.it

This file is part of MKLpy: a scikit-compliant framework for Multiple Kernel Learning
This file is distributed with the GNU General Public License v3 <http://www.gnu.org/licenses/>.  

"""

from .base import MKL, Solution
from ..arrange import summation
from sklearn.svm import SVC
import numpy as np


class Cache():
	def __init__(self, **kwargs):
		self.__dict__.update(kwargs)




class AlternateMKL(MKL):

	def __init__(self, learner=SVC(C=1000), multiclass_strategy='ova', verbose=False,
				 max_iter=1000, learning_rate=0.01, tolerance=1e-7, callbacks=[], 
				 scheduler=None, direction='min'):
		super().__init__(learner=learner, multiclass_strategy=multiclass_strategy, verbose=verbose)
		self.max_iter 		= max_iter
		self.learning_rate	= learning_rate
		self.tolerance		= tolerance
		self.callbacks		= callbacks
		self.scheduler		= scheduler
		self.direction		= direction
		assert direction in ['min']	#max not supported yet

		self.convergence 	= False
		self.cache 			= Cache()

		for callback in self.callbacks:
			callback.register(self)
		if scheduler:
			self.scheduler.register(self)


	def get_params(self, deep=True):
		# this estimator has parameters:
		new_params = {'max_iter' 		: self.max_iter,
					  'learning_rate'	: self.learning_rate,
					  'tolerance'		: self.tolerance,
					  'callbacks'		: self.callbacks,
					  }
		return super().get_params().update(new_params)



	@classmethod
	def initialize_optimization(self):
		''' this method initialize the context,
			data structures for the optimization,
			and returns the initial solution (context) '''
		raise NotImplementedError('This method has to be implemented in the derived class')
		

	@classmethod
	def do_step(self, sol):
		''' this method computes an optimization step.
			sol is the incumbent solution '''
		raise NotImplementedError('This method has to be implemented in the derived class')


	def _combine_kernels(self):
		''' the combination is overrided to allow
			an iterative optimization '''

		self.solution = self.initialize_optimization()

		for callback in self.callbacks:
			callback.on_train_begin()

		

		self.convergence = False
		step = 0
		while not self.convergence:
			step += 1

			#stop: max_iter (convergence is False)
			if step > self.max_iter:
				print ('max iter reached')
				break

			#callbacks on_step_begin
			for callback in self.callbacks:
				callback.on_step_begin(step)

			current_solution = self.do_step()
			
			improvement = self.solution.objective - current_solution.objective
			if self.scheduler:
				self.scheduler.step(step, improvement)

			# if the current solution is negative, then I can play with lr
			if improvement < 0 and self.scheduler:
				print ('[%d][AlternateMKL] negative improvement')
				continue
			

			#everything looks ok
			self.solution = current_solution

			# callbacks on_step_end
			for callback in self.callbacks:
				callback.on_step_end(step)

			if improvement < self.tolerance:
				self.convergence = True

		# callbacks optimization end
		for callback in self.callbacks:
			callback.on_train_end()

		return self.solution






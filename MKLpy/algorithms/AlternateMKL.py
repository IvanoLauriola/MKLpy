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

	
	max_iter 	= None
	lr 			= None
	tolerance 	= None
	decay 		= None
	direction	= None


	def __init__(self, learner=SVC(C=1000), generator=HPK_generator(n=10), multiclass_strategy='ova', verbose=False,
				 max_iter=1000, learning_rate=0.1, tolerance=1e-6, decay=0.8, func_form=summation, callbacks=[]):
		super().__init__(learner=learner, generator=generator, multiclass_strategy=multiclass_strategy, func_form=func_form, verbose=verbose)
		self.max_iter 		= max_iter
		self.learning_rate	= learning_rate
		self.tolerance		= tolerance
		self.decay			= decay
		self.callbacks		= callbacks

		# these values should be implemented in the derived classes
		self.direction		= None


	@classmethod
	def initialize_optimization(self):
		''' this method initialize the initial context and data structures for the optimization,
			and it returns the initial weights vector'''
		raise NotImplementedError('This method has to be implemented in the derived class')

	@classmethod
	def do_step(self, lr, sol, w, obj):
		'''this method computes an optimization step'''
		raise NotImplementedError('This method has to be implemented in the derived class')


	def _combine_kernels(self):
		# initialize callbacks
		for callback in self.callbacks:
			callback.on_train_begin()

		# initialize optimization parameters
		obj_values	= []	# values of the obj function
		stop_cond	= False
		step 		= 0		# current iteration
		lr 			= self.learning_rate
		
		obj 		= None  # current objective function
		sol			= None	# current solution, useful for reoptimization
		# initialize optimization problem
		w = self.initialize_optimization()

		##########################
		# MAIN OPTIMIZATION LOOP #
		##########################
		while not stop_cond:
			step += 1
			# callbacks on_step_begin
			for callback in self.callbacks:
				callback.on_step_begin(step)

			new_solution = self.do_step(lr, sol, w, obj)
			if not new_solution:
				break
			else:
				_obj, _w, _sol = new_solution

			improv = np.inf if not obj else obj - _obj if self.direction == 'min' else _obj - obj
			if 	improv > 0:		# correct optimization case	
				obj_values.append(_obj)
				# update the incumbent solution
				obj = _obj
				sol = _sol
				w   = _w
			else:				# worsening of the objective function
				# withdraw the current solution and reduce the learning date
				lr *= self.decay

			# check the stop conditions
			stop_step   = step > 4
			stop_improv = 0 < improv < self.tolerance
			stop_lr     = lr < 1e-7
			stop_steps  = step >= self.max_iter
			stop_cond = stop_step and (stop_improv or stop_lr or stop_steps)

			print ('+' if improv > 0 else '-', obj, improv, np.dot(w,w), lr)
			# callbacks on_step_end
			for callback in self.callbacks:
				callback.on_step_end(step)

		##########################
		# STOP OPTIMIZATION LOOP #
		##########################

		self.obj_values	= obj_values
		self.steps 		= step
		self.weights 	= w
		self.ker_matrix = self.func_form(self.KL,self.weights)

		# callbacks optimization end
		for callback in self.callbacks:
			callback.on_train_end()

		return self.ker_matrix





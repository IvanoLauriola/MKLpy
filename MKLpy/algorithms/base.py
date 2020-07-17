# -*- coding: latin-1 -*-

"""
@author: Ivano Lauriola
@email: ivano.lauriola@phd.unipd.it

This file is part of MKLpy: a scikit-compliant framework for Multiple Kernel Learning
This file is distributed with the GNU General Public License v3 <http://www.gnu.org/licenses/>.  

"""

from sklearn.utils.multiclass import check_classification_targets
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from ..arrange import average, summation
from ..utils.validation import check_KL_Y
from ..utils.exceptions import BinaryProblemError
from ..multiclass import OneVsOneMKLClassifier, OneVsRestMKLClassifier
import torch


class Solution():
	def __init__(self, weights, objective, ker_matrix, dual_coef, bias, **kwargs):
		self.weights 	= weights
		self.objective 	= objective
		self.ker_matrix = ker_matrix
		self.dual_coef 	= dual_coef
		self.bias		= bias
		self.__dict__.update(kwargs)

class Cache():
	def __init__(self, **kwargs):
		self.__dict__.update(kwargs)



class MKL(BaseEstimator, ClassifierMixin):

	# These attributes need to be set in the derived classes
	func_form  = None   # a function which takes a list of kernels and their weights and returns the combination
	n_kernels  = None   # the number of kernels used in combination
	KL 		   = None 	# the kernels list
	solution   = None 	# solution of the algorithm


	def __init__(self, 
		multiclass_strategy = 'ova',
		verbose             = False,
		tolerance           = 1e-7,
		learner             = None,
		max_iter			= -1
		):
		
		self.multiclass_strategy = multiclass_strategy # multiclass pattern ('ovo' or 'ovr')
		self.verbose     = verbose 		# logging strategy
		self.tolerance   = tolerance	# numerical tolerance
		self.learner     = learner		# the base learner which uses the combined kernel
		self.max_iter 	 = max_iter		# maximum number of iterations
		
		self.is_fitted   = False
		self.multiclass_ = None
		self.classes_    = None
		if learner:
			self.learner.kernel = 'precomputed'
		assert multiclass_strategy in ['ovo', 'ovr', 'ova'], multiclass_strategy


	def _prepare(self, KL, Y):
		'''preprocess data before training'''

		self.KL, self.Y = check_KL_Y(KL, Y)
		check_classification_targets(Y)
		self.n_kernels = len(self.KL)

		self.classes_ = self.Y.unique()
		if len(self.classes_) < 2:	# these algorithms are designed for classification only
			raise ValueError("The number of classes has to be almost 2; got ", len(self.classes_))
		self.multiclass_ = len(self.classes_) > 2
		return


	def fit(self, KL, Y):
		'''complete fit with preprocess'''
		self._prepare(KL, Y)

		self.is_fitted = True
		if self.multiclass_ :	# a multiclass wrapper is used in case of multiclass target
			metaClassifier = OneVsOneMKLClassifier if self.multiclass_strategy == 'ovo' else OneVsRestMKLClassifier
			self.clf = metaClassifier(self.__class__(**self.get_params())).fit(self.KL,self.Y)
			self.solution = self.clf.solution
		else :
			self._fit()					# fit the model

		self.is_fitted = True
		return self




	def _fit(self):
		raise NotImplementedError('This method has to be implemented in the derived class')


	def score(self, KL):
		raise NotImplementedError('This method has to be implemented in the derived class')



	def predict(self, KL):
		if not self.is_fitted :
			raise NotFittedError("The base learner is not fitted yet. Call 'fit' with appropriate arguments before using this method.")
		if self.multiclass_:
			return self.clf.predict(KL) 
		elif self.learner: 
			return self.learner.predict(self.func_form(KL,self.solution.weights))
		else:
			return torch.tensor([self.classes_[1] if p >=0 else self.classes_[0] for p in self.score(KL)])
		

	def decision_function(self, KL):
		if not self.is_fitted :
			raise NotFittedError("The base learner is not fitted yet. Call 'fit' with appropriate arguments before using this method.")
		if self.multiclass_:
			return self.clf.decision_function(KL) 
		elif self.learner: 
			return self.learner.decision_function(self.func_form(KL,self.solution.weights))
		else:
			return self.score(KL)




	def set_params(self, **parameters):
		for parameter, value in parameters.items():
			setattr(self,parameter,value)
		return self

	def get_params(self,deep=True):
		return {"learner":self.learner,
				"verbose":self.verbose,
				"multiclass_strategy":self.multiclass_strategy,
				"tolerance": self.tolerance,
				"max_iter": self.max_iter,
				}

	def score(self, KL):
		Kte = self.func_form(KL, self.solution.weights)
		ygamma = self.solution.dual_coef.T * torch.tensor([1 if y==self.classes_[1] else -1 for y in self.Y])
		return Kte @ ygamma - self.solution.bias





class OneStepMKL(MKL):
	
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		assert self.learner is not None

	def _fit(self):
		self.solution = self._combine_kernels()	# call combine_kernels without re-preprocess
		self.learner.fit(self.solution.ker_matrix,self.Y)	# fit the base learner
		return

	def combine_kernels(self, KL, Y=None):
		'''only kernels combination, with preprocess'''
		self._prepare( KL, Y)
		if self.multiclass_:
			raise BinaryProblemError("combine_kernels requires binary classification problems")
		self.solution = self._combine_kernels()
		return self.solution.ker_matrix

	def _combine_kernels(self):
		'''implemented in base class, return a kernel'''
		raise NotImplementedError('This method has to be implemented in the derived class')





class TwoStepMKL(MKL):

	direction 	= None
	convergence = False
	cache 		= Cache()

	def __init__(self, learning_rate=.01, callbacks=[], scheduler=None, max_iter=1000, **kwargs):
		super().__init__(max_iter=max_iter, **kwargs)
		self.learning_rate 	= learning_rate
		self.callbacks 		= callbacks
		self.scheduler 		= scheduler
		self.initial_lr 	= learning_rate

		for callback in self.callbacks:
			callback.register(self)
		if scheduler:
			self.scheduler.register(self)


	def get_params(self, deep=True):
		params = super().get_params()
		params.update({
			'learning_rate'	: self.initial_lr,
			'callbacks'		: self.callbacks,
			'scheduler'     : self.scheduler,
		})
		return params


	def initialize_optimization(self):
		''' this method initialize the context,
			data structures for the optimization,
			and returns the initial solution (context) '''
		raise NotImplementedError('This method has to be implemented in the derived class')


	def do_step(self, sol):
		''' this method computes an optimization step abd returns the new solution.
			sol is the current solution '''
		raise NotImplementedError('This method has to be implemented in the derived class')


	def _fit(self):

		self.learning_rate = self.initial_lr
		self.solution = self.initialize_optimization()
		for callback in self.callbacks: callback.on_train_begin()
		self.convergence = False
		multiplier = 1 if self.direction=='min' else -1

		step = 0
		while not self.convergence and step < self.max_iter:

			step += 1
			for callback in self.callbacks: callback.on_step_begin(step)

			# 1) next optimization step
			current_solution = self.do_step()

			# 2) check improvement and LR
			improvement = self.solution.objective - current_solution.objective
			improvement *= multiplier
			if self.scheduler: 
				self.convergence = self.convergence or self.scheduler.step(step, improvement)
				if improvement < 0:
					continue
			
			# 3) if everything looks ok, update the solution
			self.solution = current_solution

			# on_step_end is invoked only when the current step is saved
			for callback in self.callbacks: callback.on_step_end(step)
			#if improvement < self.tolerance: self.convergence = True
			# end cycle

		for callback in self.callbacks: callback.on_train_end()

		if self.learner:
			self.learner.fit(self.solution.ker_matrix, self.Y)

		return self.solution


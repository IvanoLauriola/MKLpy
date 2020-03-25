# -*- coding: latin-1 -*-

"""
@author: Ivano Lauriola
@email: ivano.lauriola@phd.unipd.it

This file is part of MKLpy: a scikit-compliant framework for Multiple Kernel Learning
This file is distributed with the GNU General Public License v3 <http://www.gnu.org/licenses/>.  

"""

from sklearn.utils.multiclass import check_classification_targets
from sklearn.base import BaseEstimator, ClassifierMixin
from ..arrange import average, summation
from ..utils.validation import check_KL_Y
from ..utils.exceptions import BinaryProblemError
from ..multiclass import OneVsOneMKLClassifier, OneVsRestMKLClassifier
import numpy as np


class Solution():
	def __init__(self, weights, objective, ker_matrix, **kwargs):
		self.weights 	= weights
		self.objective 	= objective
		self.ker_matrix = ker_matrix
		self.__dict__.update(kwargs)


class MKL(BaseEstimator, ClassifierMixin):

	func_form  = None   # a function which taskes a list of kernels and their weights and returns the combination
	n_kernels  = None   # the number of kernels used in combination
	KL 		   = None 	# the kernels list
	solution   = None 	# solution of the algorithm


	def __init__(self, learner, multiclass_strategy, verbose):
		self.learner     = learner		# the base learner which uses the combined kernel
		self.verbose     = verbose 		# logging strategy
		self.multiclass_strategy = multiclass_strategy # multiclass pattern ('ovo' or 'ovr')

		self.is_fitted   = False
		self.multiclass_ = None
		self.classes_    = None
		self.learner.kernel = 'precomputed'
		assert multiclass_strategy in ['ovo', 'ovr', 'ova']


	def _prepare(self, KL, Y):
		'''preprocess data before training'''
		check_classification_targets(Y)
		self.classes_ = np.unique(Y)
		if len(self.classes_) < 2:	# these algorithms are meant for classification only
			raise ValueError("The number of classes has to be almost 2; got ", len(self.classes_))
		self.multiclass_ = len(self.classes_) > 2

		self.KL, self.Y = check_KL_Y(KL,Y)
		self.n_kernels = len(self.KL)
		return


	def fit(self, KL, Y):
		'''complete fit with preprocess'''
		self._prepare(KL, Y)

		if self.multiclass_ :	# a multiclass wrapper is used in case of multiclass target
			metaClassifier = OneVsOneMKLClassifier if self.multiclass_strategy == 'ovo' else OneVsRestMKLClassifier
			self.clf = metaClassifier(self.__class__(**self.get_params())).fit(self.KL,self.Y)
			self.solution = self.clf.solution
		else :
			self._fit()					# fit the model
		self.is_fitted = True
		return self


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
		return solution.ker_matrix


	def _combine_kernels(self):
		'''implemented in base class, return a kernel'''
		raise NotImplementedError('This method has to be implemented in the derived class')



	def predict(self, KL):
		if not self.is_fitted :
			raise NotFittedError("The base learner is not fitted yet. Call 'fit' with appropriate arguments before using this method.")
		return self.clf.predict(KL) if self.multiclass_ else self.learner.predict(self.func_form(KL,self.solution.weights))
		

	def decision_function(self, KL):
		if self.is_fitted == False:
			raise NotFittedError("The base learner is not fitted yet. Call 'fit' with appropriate arguments before using this method.")
		if self.multiclass_:
			raise ValueError('Scores are not available for multiclass problems, use predict')
		return self.learner.decision_function(self.func_form(KL,self.solution.weights))



	def set_params(self, **parameters):
		for parameter, value in parameters.items():
			setattr(self,parameter,value)
		return self

	def get_params(self,deep=True):
		return {"learner":self.learner,
				"verbose":self.verbose,
				"multiclass_strategy":self.multiclass_strategy}









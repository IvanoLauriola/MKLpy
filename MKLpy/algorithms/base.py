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
from ..lists.generator import HPK_generator
from ..utils.validation import process_list, check_KL_Y
from ..multiclass import OneVsOneMKLClassifier, OneVsRestMKLClassifier
import numpy as np

class MKL(BaseEstimator, ClassifierMixin):

	func_form  = None   # the functional form in combination
	ker_matrix = None   # the obtained kernel matrix
	weights    = None   # the weights used in combination
	learner    = None   # the base learner
	n_kernels  = None   # the number of kernels used in combination
	generator  = None   # the generator of kernels
	KL 		   = None 	# the kernels list

	def __init__(self, learner, generator, func_form, multiclass_strategy, verbose):
		self.learner     = learner
		self.generator   = generator
		self.func_form   = func_form
		self.verbose     = verbose
		self.is_fitted   = False
		self.multiclass_ = None
		self.classes_    = None
		self.weights     = None
		self.multiclass_strategy = multiclass_strategy
		self.learner.kernel = 'precomputed'


	def _prepare(self,X,Y):
		'''preprocess data before training'''
		check_classification_targets(Y)
		self.classes_ = np.unique(Y)
		if len(self.classes_) < 2:
			raise ValueError("The number of classes has to be almost 2; got ", len(self.classes_))
		self.multiclass_ = len(self.classes_) > 2

		KL = process_list(X,self.generator)				# X can be a samples matrix or Kernels List
		self.KL, self.Y = check_KL_Y(KL,Y)
		self.n_kernels = len(self.KL)
		return

	def fit(self,X,Y):
		'''complete fit with preprocess'''
		self._prepare(X,Y)

		if self.multiclass_ :
			metaClassifier = OneVsOneMKLClassifier if self.multiclass_strategy in ['ovo','OvO','1v1'] else OneVsRestMKLClassifier
			self.clf = metaClassifier(self.__class__(**self.get_params())).fit(self.KL,self.Y)
			self.weights = self.clf.weights
			self.ker_matrix = self.clf.ker_matrices
		else :
			self._fit()					# fit the model

		self.is_fitted = True
		return self

	def _fit(self):
		self.ker_matrix = self._combine_kernels()	# call combine_kernels without re-preprocess
		self.learner.fit(self.ker_matrix,self.Y)					# fit model using the base learner
		return

	def combine_kernels(self,X,Y=None):
		'''only kernels combination, with preprocess'''
		self._prepare(X,Y)
		if self.multiclass_:
			raise ValueError("combine_kernels requires binary classification problems")
		return self._combine_kernels()


	def _combine_kernels(self):
		'''implemented in base class, return a kernel'''
		raise NotImplementedError('This method has to be implemented in the derived class')



	def predict(self,X):
		if not self.is_fitted :
			raise NotFittedError("The base learner is not fitted yet. Call 'fit' with appropriate arguments before using this method.")
		KL = process_list(X,self.generator)
		return self.clf.predict(KL) if self.multiclass_ else self.learner.predict(self.func_form(KL,self.weights))
		

	def decision_function(self,X):
		if self.is_fitted == False:
			raise NotFittedError("The base learner is not fitted yet. Call 'fit' with appropriate arguments before using this method.")
		if self.multiclass_:
			raise ValueError('Scores are not available for multiclass problems, use predict')
		KL = process_list(X,self.generator)				# X can be a samples matrix or Kernel List
		return self.learner.decision_function(self.func_form(KL,self.weights))



	def set_params(self, **parameters):
		for parameter, value in parameters.items():
			setattr(self,parameter,value)
		return self

	def get_params(self,deep=True):
		raise NotImplementedError('This method has to be implemented in the derived class')











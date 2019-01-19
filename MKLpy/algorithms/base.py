from sklearn.svm import SVC
from sklearn.utils.multiclass import check_classification_targets
from MKLpy.arrange import average
from MKLpy.lists.generator import HPK_generator
from MKLpy.utils.validation import process_list, check_KL_Y
from MKLpy.multiclass import OneVsOneMKLClassifier, OneVsRestMKLClassifier
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class MKL(object):

	how_to     = None   # the functional form in combination
	ker_matrix = None   # the obtained kernel matrix
	weights    = None   # the weights used in combination
	estimator  = None   # the base learner
	n_kernels  = None   # the number of kernels used in combination


	KL 		   = None 	# the kernels list
	generator  = None 	# 

	def __init__(self,estimator,generator,how_to,multiclass_strategy,max_iter,verbose):
		self.estimator = estimator
		self.generator = generator
		self.how_to = how_to
		self.multiclass_strategy = multiclass_strategy
		self.max_iter = max_iter
		self.verbose = verbose
		self.estimator.kernel='precomputed'
		self.is_fitted = False
		self.multiclass_ = None
		self.classes_ = None
		self.weights = None


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
		self.ker_matrix = self._arrange_kernel()	# call arrange_kernel without re-preprocess
		self.estimator.fit(self.ker_matrix,self.Y)					# fit model using the base learner
		return

	def arrange_kernel(self,X,Y=None):
		'''only kernels combination, with preprocess'''
		self._prepare(X,Y)
		if self.multiclass_:
			raise ValueError("arrange_kernel requires binary classification problems")
		return self._arrange_kernel()


	def _arrange_kernel(self,KL,Y):
		'''implemented in base class, return a kernel'''
		raise NotImplementedError('Not implemented yet')



	def predict(self,X):
		if not self.is_fitted :
			raise NotFittedError("This KOMD instance is not fitted yet. Call 'fit' with appropriate arguments before using this method.")
		KL = process_list(X,self.generator)
		return self.clf.predict(KL) if self.multiclass_ else self.estimator.predict(self.how_to(KL,self.weights))
		#return self.estimator.decision_function(self.how_to(KL,self.weights))

	def decision_function(self,X):
		if self.is_fitted == False:
			raise NotFittedError("This KOMD instance is not fitted yet. Call 'fit' with appropriate arguments before using this method.")
		if self.multiclass_:
			raise ValueError('Scores are not available for multiclass problems, use predict')
		KL = process_list(X,self.generator)				# X can be a samples matrix or Kernel List
		return self.estimator.decision_function(self.how_to(KL,self.weights))



	def set_params(self, **parameters):
		for parameter, value in parameters.items():
			setattr(self,parameter,value)
		return self

	def get_params(self,deep=True):
		raise NotImplementedError('Not implemented yet')




class AverageMKL(BaseEstimator, ClassifierMixin, MKL):

	def __init__(self, estimator=SVC(C=1), generator=HPK_generator(n=10), multiclass_strategy='ova', max_iter=100, verbose=False):
		super(self.__class__, self).__init__(estimator=estimator, generator=generator, multiclass_strategy=multiclass_strategy, how_to=average, max_iter=max_iter, verbose=verbose)
		#set other params


	def _arrange_kernel(self):
		self.weights = np.ones(self.n_kernels)/self.n_kernels	# weights vector is a np.ndarray
		ker_matrix = self.how_to(self.KL,self.weights)		# combine kernels
		return ker_matrix

	def get_params(self, deep=True):
		return {"estimator":self.estimator,
				"generator":self.generator,
				"verbose":self.verbose,
				"multiclass_strategy":self.multiclass_strategy,
				"max_iter":self.max_iter}






from sklearn.metrics  import completeness_score, v_measure_score, homogeneity_score
import numpy as np

class Binarizator():
	''' base class for binarization algorithms '''
	def fit(self,X,Y):
		raise NotImplementedError('Not implemented yet')

	def transform(self,X):
		raise NotImplementedError('Not implemented yet')

	def get_params(self,deep=True):
		raise NotImplementedError('Not implemented yet')

	def set_params(self, **parameters):
		for parameter, value in parameters.items():
			setattr(self,parameter,value)
		return self


class AverageBinarizator(Binarizator):
	''' performs a feature discretization, 1 if the value is over an average threshold, 0 else '''
	def __init__(self):
		return

	def fit (self,X,Y):
		self.cols = np.average(X,axis=0)
		return self

	def transform(self,X):
		return (X > self.cols) * 1

	def get_params(self,deep=True):
		return {}



class EntropyBinarizator(Binarizator):

	def __init__(self):
		pass

	def fit(self,X,Y):
		n,m = X.shape[0],X.shape[1]
		self.cols = np.zeros(m)
		for i in xrange(m):
			v = np.unique(X[:,i])
			v.sort()

			top = {'e':None, 's':None}
			for j in xrange(len(v)-1):
				s = v[j:j+2].mean()
				e = completeness_score(Y,[1 if y < s else 0 for y in Y])
				if not top['e'] or e > top['e']:
					top = {'e':e, 's':s}
			self.cols[i] = top['s']
		return self

	def transform(self,X):
		return (X > self.cols) * 1

	def get_params(self,deep=True):
		return {}
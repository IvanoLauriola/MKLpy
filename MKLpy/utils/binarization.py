

from sklearn.metrics  import completeness_score, v_measure_score, homogeneity_score
import numpy as np

class Binarizator(Object):

	def __init__():
		pass

	def transform(X,Y=None):
		pass

class AverageBinarizator(Binarizator):

	def __init__():
		return

	def transform(X,Y=None):
		cols = np.average(X,axis=0)
		return np.concatenate((X>cols,X<=cols),axis=1)

class EntropyBinarizator(Binarizator):

	def __init__():
		return

	def transform(X,Y):
		n,m = X.shape[0],X.shape[1]
		cols = np.zeros(m)
		for i in xrange(m):
			v = np.unique(X[:,i])
			v.sort()

			top = {'e':None, 's':None}
			for j in xrange(len(v)-1):
				s = v[j:j+2].mean()
				e = completeness_score(Y,[1 if y < s else 0 for y in Y])
				if not top['e'] or e > top['e']:
					top = {'e':e, 's':s}
			cols[i] = top['s']
		return np.concatenate((X>cols,X<=cols),axis=1)
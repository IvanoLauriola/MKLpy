

from sklearn.metrics  import completeness_score, v_measure_score, homogeneity_score
import numpy as np
from math import log

class Binarizer():
	''' base class for binarization algorithms '''
	def fit(self,X,Y):
		raise NotImplementedError('Not implemented yet')

	def fit_transform(self,X,Y):
		self.fit(X,Y)
		return self.transform(X)

	def transform(self,X):
		raise NotImplementedError('Not implemented yet')

	def get_params(self,deep=True):
		raise NotImplementedError('Not implemented yet')

	def set_params(self, **parameters):
		for parameter, value in parameters.items():
			setattr(self,parameter,value)
		return self


class AverageBinarizer(Binarizer):
	''' performs a feature discretization, 1 if the feature is over the average value threshold, 0 else '''

	def fit (self,X,Y):
		self.cols = np.average(X,axis=0)
		return self

	def transform(self,X):
		return np.concatenate((X>self.cols,X<=self.cols),axis=1) * 1

	def get_params(self,deep=True):
		return {}



class EntropyBinarizer(Binarizer):

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
		return np.concatenate((X>self.cols,X<=self.cols),axis=1) * 1

	def get_params(self,deep=True):
		return {}



class MDLPBinarizer(Binarizer):

	def __entropy(self, y):
		N = float(len(y))
		ent = 0
		for c in set(y):
			proportion = len(y[y == c]) / N
			ent -= proportion * log(proportion, 2)
		return ent


	def __get_id_by_condition(self, X, f, cond):
		return np.array([i for i in range(X.shape[0]) if cond(X[i,f])])


	def __cut_point_ig(self, X, y, feature, cut_point):
		X_left  = self.__get_id_by_condition(X, feature, lambda f : f <= cut_point)
		X_right = self.__get_id_by_condition(X, feature, lambda f : f >  cut_point)
	
		return self.__entropy(y) - 1./ float(X.shape[0]) * (len(X_left ) * self.__entropy(y[X_left] ) \
								 						  + len(X_right) * self.__entropy(y[X_right]))
				

	def criterion(self, XX, yy, feature, cut_point):
		X_left  = self.__get_id_by_condition(XX, feature, lambda f : f <= cut_point)
		X_right = self.__get_id_by_condition(XX, feature, lambda f : f > cut_point)
		y_left  = yy[X_left]
		y_right = yy[X_right]

		N, k = XX.shape[0], len(set(yy))
		delta = log(3 ** k, 2) - (k * self.__entropy(yy)) + (len(set(y_left)) * self.__entropy(y_left)) + \
		 													(len(set(y_right)) * self.__entropy(y_right))
		
		return self.__cut_point_ig(XX, yy, feature, cut_point) > (log(N - 1, 2) + delta) / N


	def feature_boundaries(self, feature):
		np_sorted = np.argsort(self.X[:,feature])
		sorted_idx = list(np_sorted)
		X_ord = self.X[sorted_idx,feature]

		f_offset = np.roll(X_ord, 1)
		f_change = (X_ord != f_offset)
		mid_points = (X_ord + f_offset) / 2.
		potential_cuts = np_sorted[(np.where(f_change == True)[0])[1:]]

		boundary_points = []
		for row in potential_cuts:
			old = sorted_idx[sorted_idx.index(row) - 1] 
			old_classes = set(self.y[np.where(self.X[:,feature] == self.X[old, feature])])
			new_classes = set(self.y[np.where(self.X[:,feature] == self.X[row, feature])])
			if len(set.union(set(old_classes), set(new_classes))) > 1:
				boundary_points.append(mid_points[sorted_idx.index(row)])

		return set(boundary_points)


	def get_boundaries(self, X, feature):
		range_min, range_max = (X[:,feature].min(), X[:,feature].max())
		return set([x for x in self._boundaries[feature] if (x > range_min) and (x < range_max)])


	def best_cut_point(self, X, y, feature):
		candidates = self.get_boundaries(X, feature)
		if candidates:
			gains = [(cut, self.__cut_point_ig(X, y, feature, cut)) for cut in candidates]
			gains = sorted(gains, key=lambda x: x[1], reverse=True)
			return gains[0][0]


	def feature_cutpoints(self, feature, part_index):		
		XX = self.X[part_index,:]
		yy = self.y[part_index]
		
		if set(XX[:,feature]) >= 2:
			candidate = self.best_cut_point(XX, yy, feature)
		
			if candidate:
				if self.criterion(XX, yy, feature, candidate):
					left_part  = XX[:,feature][XX[:,feature] <= candidate]
					right_part = XX[:,feature][XX[:,feature] >  candidate]
					part_index = np.array(part_index)
					
					X_left  = part_index[self.__get_id_by_condition(XX, feature, lambda f : f <= candidate)]
					X_right = part_index[self.__get_id_by_condition(XX, feature, lambda f : f >  candidate)]
					
					if X_left.size > 0 and X_right.size > 0:
						self._cuts[feature].append(candidate)
						self.feature_cutpoints(feature, X_left)
						self.feature_cutpoints(feature, X_right)
						self._cuts[feature] = sorted(self._cuts[feature])
	
	
	def fit(self, X, Y):
		self._features = [i for i in range(X.shape[1]) if len(set(X[:,i])) > 5] #FIXME
		self.y = Y
		self.X = X[:,self._features]
		self._ignored_features = set(range(X.shape[1])) - set(self._features)
		self._boundaries = {f: self.feature_boundaries(f) for f in self._features}
		self._cuts = {f: [] for f in self._features}
		
		for feature in self._features:
			self.feature_cutpoints(feature, range(self.X.shape[0]))
		return self


	def transform(self, X):
		for i in self._cuts:
			self._cuts[i].insert(0, -np.inf)
			self._cuts[i].append(np.inf)
			
		n_bin_features = len(self._ignored_features) + sum([len(s)-1 for s in self._cuts.values()])
		X_new = np.zeros((X.shape[0], n_bin_features))
		
		n_orig_features = len(self._ignored_features) + len(self._features)
		b = 0
		for orig in range(n_orig_features):
			if orig in self._ignored_features:
				X_new[:,b] = X[:,orig]
				b += 1
			else:
				for j in range(len(self._cuts[orig])-1):
					condlist = [self._cuts[orig][j+1] < X[:,orig], self._cuts[orig][j] < X[:,orig]]
					X_new[:,b] = np.select(condlist, [0,1])
					b += 1
					
		return X_new


	def get_params(self, deep=True):
		return {}
		
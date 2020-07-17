# -*- coding: latin-1 -*-

"""
@author: Ivano Lauriola
@email: ivano.lauriola@phd.unipd.it, ivanolauriola@gmail.com

This file is part of MKLpy: a scikit-compliant framework for Multiple Kernel Learning
This file is distributed with the GNU General Public License v3 <http://www.gnu.org/licenses/>.  

"""

import numpy as np
from sklearn.exceptions import NotFittedError
from itertools import combinations
from . import algorithms #import algorithms.base



class MulticlassMKLClassifier():
	def __init__(self, mkl, verbose=False):
		self.mkl = mkl
		self.learner = mkl.learner
		self.verbose = verbose
		self.is_fitted = False
		self.classes_ = None


	def _generate_tasks(self, labels):
		raise NotImplementedError('This method has to be implemented in the derived class')


	def fit(self, KL, Y):
		self.classes_ = np.unique(Y)
		self.tasks = self._generate_tasks(Y)

		self.estimators_ = {}
		self.solution = {}
		for task in self.tasks:
			mkl = self.mkl.__class__(**self.mkl.get_params())
			mkl.learner = mkl.learner.__class__(**mkl.learner.get_params())
			mkl.learner.kernel = 'precomputed'

			idx_pos = self.tasks[task]['idx_pos']
			idx_neg = self.tasks[task]['idx_neg']
			all_idx = idx_pos + idx_neg
			KLt = [K[all_idx][:,all_idx] for K in KL]	#todo: fix for generators
			Yt  = [1]*len(idx_pos) + [-1]*len(idx_neg)

			mkl = mkl.fit(KLt, Yt)
			self.estimators_[task] = mkl
			self.solution[task] = mkl.solution

		self.is_fitted = True
		return self


	def _get_scores(self, KL):
		if not self.is_fitted:
			raise NotFittedError("The base learner is not fitted yet. Call 'fit' with appropriate arguments before using this method.")
		preds, scores = {}, {}
		for task, mkl in self.estimators_.items():
			all_idx = self.tasks[task]['idx_pos'] + self.tasks[task]['idx_neg']
			KLt = [K[:, all_idx] for K in KL]

			scores[task] = mkl.decision_function(KLt)
			preds[task]  = mkl.predict(KLt)
		return scores, preds


	def predict(self, KL):
		scores, preds = self._get_scores(KL)
		return self.voting(scores, preds)
	

	def decision_function(self, KL):
		scores, preds = self._get_scores(KL)
		return scores

	def voting(self, scores):
		raise NotImplementedError('This method has to be implemented in the derived class')



class OneVsRestMKLClassifier(MulticlassMKLClassifier):

	def _generate_tasks(self, Y):
		tasks = {}
		for l in self.classes_:
			idx_pos = [i for i,y in enumerate(Y) if y==l]
			idx_neg = [i for i,y in enumerate(Y) if y!=l]
			tasks[l] = {'idx_pos': idx_pos, 'idx_neg': idx_neg}
		return tasks

	def voting(self, scores, preds):
		nn = len(next(iter(scores.values())))
		y_pred = [max({key:scores[key][i] for key in scores}.items(), key=lambda x:x[1])[0] \
			for i in range(nn)]
		return np.array(y_pred)



class OneVsOneMKLClassifier(MulticlassMKLClassifier):

	def _generate_tasks(self, Y):
		tasks = {}
		for cp, cn in combinations(self.classes_, 2):
			idx_pos = [i for i,y in enumerate(Y) if y==cp]
			idx_neg = [i for i,y in enumerate(Y) if y==cn]
			tasks[(cp,cn)] = {'idx_pos': idx_pos, 'idx_neg': idx_neg}
		return tasks

	def voting(self, scores, preds):
		nn = len(next(iter(preds.values())))
		points = np.zeros((nn, len(self.classes_)), dtype=int)
		print (points.shape)

		labels_to_id = {c:i for i,c in enumerate(self.classes_)}
		for task in preds.keys():
			cp, cn = labels_to_id[task[0]], labels_to_id[task[1]]
			for iv,v in enumerate( preds[task] ):
				points[iv, cp if v==1 else cn] += 1

		y_pred = np.argmax(points,1)
		return y_pred

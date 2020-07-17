# -*- coding: latin-1 -*-

"""
@author: Ivano Lauriola
@email: ivano.lauriola@phd.unipd.it, ivanolauriola@gmail.com

This file is part of MKLpy: a scikit-compliant framework for Multiple Kernel Learning
This file is distributed with the GNU General Public License v3 <http://www.gnu.org/licenses/>.  

"""

import numpy as np
import warnings
from .algorithms.base import Solution
from .utils.validation import get_scorer


class Callback():

	def __init__(self):
		self.model = None

	def register(self, model):
		self.model = model

	def on_train_begin(self):
		pass

	def on_train_end(self):
		pass

	def on_step_begin(self, step):
		pass

	def on_step_end(self, step):
		pass



class Monitor(Callback):

	def __init__(self, metrics=[]):
		super().__init__()
		self.metrics = metrics

	def on_train_begin(self):
		self.objective = []
		self.weights   = []
		self.history   = {metric.__name__: [] for metric in self.metrics}

	def on_step_end(self, step):
		self.objective.append(self.model.solution.objective)
		self.weights.append(self.model.solution.weights)
		for metric in self.metrics:
			self.history[metric.__name__].append( metric(self.model.solution.ker_matrix, self.model.Y) )



class EarlyStopping(Callback):

	def __init__(self,
				KLva,
				Yva,
				patience=5,
				cooldown=1,
				metric='roc_auc',
				restore_best_solution=True,
			):

		super().__init__()
		self.KLva = KLva
		self.Yva  = Yva
		self.patience = patience
		self.cooldown = cooldown
		self.metric = metric
		self.restore_best_solution = restore_best_solution

	def on_train_begin(self):
		self.wait = 0
		self.stopped_epoch = 0
		self.best_solution = self.model.solution
		self.vals = []

		self.scorer, f, self.monitor_op = get_scorer(self.metric, return_direction=True)
		self.f = getattr(self.model, f)
		self.best = -np.Inf if self.monitor_op == np.greater else np.Inf

	def on_step_end(self, step):
		if step % self.cooldown:
			return
		
		ys = self.f(self.KLva)
		current = self.scorer(self.Yva, ys)
		self.vals.append(current)


		if self.monitor_op(current, self.best) or current == self.best:
			self.best_solution = Solution(
				weights = self.model.solution.weights,
				objective = self.model.solution.objective,
				ker_matrix = self.model.solution.ker_matrix,
				dual_coef = self.model.solution.dual_coef,
				bias = self.model.solution.bias,
				)
			self.best = current
			self.wait = 0
		else:
			self.wait += 1
			if self.wait >= self.patience:
				self.stoppend_epoch = step
				self.model.convergence = True

	def on_train_end(self):
		if self.restore_best_solution:
			self.model.solution = self.best_solution






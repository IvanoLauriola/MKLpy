# -*- coding: latin-1 -*-

"""
@author: Ivano Lauriola
@email: ivano.lauriola@phd.unipd.it, ivanolauriola@gmail.com

This file is part of MKLpy: a scikit-compliant framework for Multiple Kernel Learning
This file is distributed with the GNU General Public License v3 <http://www.gnu.org/licenses/>.  

"""

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
import warnings
from .algorithms.base import Solution
from .metrics import ratio


class Scheduler():

	def __init__(self):
		self.model = None

	def register(self, model):
		self.model = model

	def step(self, i, delta):
		'''checks the improvemen and adjusts, in case, the learning rate
			returns True if the convergence is reached'''
		raise NotImplementedError("This is an abstract method")
	



class ReduceOnWorsening(Scheduler):

	def __init__(self, multiplier=.5, min_lr=1e-7):
		super().__init__()
		self.multiplier = multiplier
		self.min_lr = min_lr


	def step(self, i, delta):

		if delta < 0:
			self.model.learning_rate *= self.multiplier
		return self.model.learning_rate < self.min_lr



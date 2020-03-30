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
		raise NotImplementedError("This is an abstract method")



class ReduceOnWorsening(Scheduler):

	def __init__(self, multiplier=.5, min_lr=1e-7):
		super().__init__()
		self.multiplier = multiplier
		self.min_lr = min_lr


	def step(self, i, delta):
		if delta < 0:
			self.model.learning_rate *= self.multiplier
			#print ('[%d][scheduler] %f improv. learning rate set to' % (i,delta), self.model.learning_rate)
		if self.model.learning_rate < self.min_lr:
			self.model.convergence = True
			#print ('[%d][scheduler] min lr, convergence is reached' % i)



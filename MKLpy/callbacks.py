import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
import warnings
from .algorithms.base import Solution
from .metrics import ratio


class Callback():

	def __init__(self):
		self.model = None

	def register(self, model):
		self.model = model

	def on_train_begin(self, KL, Y):
		pass

	def on_train_end(self):
		pass

	def on_step_begin(self, step):
		pass

	def on_step_end(self, step):
		pass


class EarlyStopping(Callback):

	def __init__(self,
				KLva,
				Yva,
				patience=5,
				cooldown=1,
				metric='val_obj',
				restore_best_solution=True,
			):

		super().__init__()
		self.KLva = KLva
		self.Yva  = Yva
		self.patience = patience
		self.cooldown = cooldown
		self.metric = metric
		self.restore_best_solution = restore_best_solution
		assert metric in ['obj', 'val_obj', 'auc', 'accuracy']

	def register(self, model):
		self.model = model
		if self.metric in ['auc', 'accuracy'] or model.direction=='max':
			self.monitor_op = np.greater
			self.best = -np.Inf
		else:
			self.monitor_op = np.less
			self.best = np.Inf


	def on_train_begin(self):
		self.wait = 0
		self.stopped_epoch = 0
		self.baseline = None
		self.best_solution = self.model.solution #initial sol
		#todo: set self.best
		#self.on_step_end(0)
		self.vals = []#ratio(self.model.solution.ker_matrix, self.model.Y)]
		#print ('[0][earlystopping] initial obj', self.model.solution.objective, self.vals[0])


	def on_step_end(self, step):
		if step % self.cooldown:
			return

		if self.metric == 'obj':
			current = self.model.solution.objective
		elif self.metric == 'val_obj':
			pass
		elif self.metric == 'auc':
			Kva = self.model.func_form(self.KLva, self.model.solution.weights)
			ys = self.model.learner.fit(self.model.solution.ker_matrix, self.model.Y) \
				.decision_function(Kva)
			current = roc_auc_score(self.Yva, ys)
		elif self.metric == 'accuracy':
			Kva = self.model.func_form(self.KLva, self.model.solution.weights)
			ys = self.model.learner.fit(self.model.solution.ker_matrix, self.model.Y) \
				.predict(Kva)
			current = accuracy_score(self.Yva, ys)
		else:
			raise Error('Metric error')

		self.vals.append(current)#ratio(self.model.solution.ker_matrix, self.model.Y))
		print ('[%d][earlystopping] current' % step, \
			self.model.solution.objective, self.vals[-1], current)


		if self.monitor_op(current, self.best):
			self.best_solution = Solution(
				weights = self.model.solution.weights,
				objective = self.model.solution.objective,
				ker_matrix = self.model.solution.ker_matrix,
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






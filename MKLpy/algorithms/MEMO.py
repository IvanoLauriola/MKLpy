# -*- coding: latin-1 -*-

"""
@author: Ivano Lauriola
@email: ivano.lauriola@phd.unipd.it

This file is part of MKLpy: a scikit-compliant framework for Multiple Kernel Learning
This file is distributed with the GNU General Public License v3 <http://www.gnu.org/licenses/>.  

"""

from .AlternateMKL import AlternateMKL
from ..arrange import average, summation
from ..metrics import frobenius
from sklearn.svm import SVC
import numpy as np
from cvxopt import spdiag,matrix,solvers


class MEMO(AlternateMKL):


	def __init__(self, learner=SVC(C=1000), multiclass_strategy='ova', verbose=False,
				theta=0.0, max_iter=1000, learning_rate=0.01, callbacks=[]):
		super().__init__(learner=learner, generator=generator, multiclass_strategy=multiclass_strategy,	
			max_iter=max_iter, verbose=verbose, callbacks=callbacks)
		self.theta = theta
		self.func_form = summation
		print ('warning: MEMO needs refactoring and parameters chehck, please contact the author if you want to use MEMO')

	def get_params(self, deep=True):
		new_params = {'theta': self.theta}
		params = super().get_params()
		params.update(new_params)
		return params



	def initialize_optimization(self):
		Q = np.array([[np.dot(self.KL[r].ravel(),self.KL[s].ravel()) for r in range(self.n_kernels)] for s in range(self.n_kernels)])
		Q /= np.sum([frobenius(K)**2 for K in self.KL])

		context = 	{
				'Q'		: Q,
				'YY'	: spdiag([1 if y==self.Y[0] else -1 for y in self.Y])
			}
		incumbent_solution = {'gamma': opt_margin(average(self.KL), context['YY'], None)[2]}
		weights = np.ones(self.n_kernels)/self.n_kernels
		ker_matrix = self.func_form(self.KL, weights)
		obj = np.inf
		return obj, incumbent_solution, weights, ker_matrix, context


	def do_step(self):
		Kc = self.ker_matrix
		YY = self.context['YY']
		Q  = self.context['Q' ]
		'''
		try:
			_margin, _gamma, _sol = opt_margin(Kc, YY, sol)
		except:
			return obj, w, None
		grad = np.array([(self.theta * np.dot(Q[r],w) + (_gamma.T * YY * matrix(self.KL[r]) * YY * _gamma)[0]) \
				* w[r] * (1 - w[r]) 	for r in range(self.n_kernels)])

		_beta = np.log(w) + lr * grad
		_w = np.exp(_beta)
		#_w = w + lr * grad
		_w /= _w.sum()
		_comp = np.dot(_w,np.dot(_w,Q))
		_obj = _margin + (self.theta/2) * _comp

		print(_margin,_comp)

		return _obj, _w, _sol
		'''
		w = self.weights[:]
		_gamma = self.incumbent_solution['gamma']['x']
		grad = np.array([(self.theta * np.dot(Q[r],w) + (_gamma.T * YY * matrix(self.KL[r]) * YY * _gamma)[0]) \
				* w[r] * (1 - w[r]) 	for r in range(self.n_kernels)])
		_beta = np.log(w) + self.learning_rate * grad
		_w = np.exp(_beta)
		_w /= sum(_w)
		_margin, _gamma, _sol = opt_margin(Kc, YY, _gamma)

		ker_matrix = self.func_form(self.KL, _w)
		if _margin < 1e-4:
			return self.obj, self.incumbent_solution, self.weights, self.ker_matrix

		_comp = np.dot(_w,np.dot(_w,Q))
		_obj = _margin + (self.theta/2) * _comp
		incumbent_solution = {'gamma': _sol}
		return _obj, incumbent_solution, _w, ker_matrix


def opt_margin(K,YY,init_sol=None):
	'''optimized margin evaluation'''
	n = K.shape[0]
	P = 2 * (YY * matrix(K) * YY)
	p = matrix([0.0]*n)
	G = -spdiag([1.0]*n)
	h = matrix([0.0]*n)
	A = matrix([[1.0 if YY[i,i]==+1 else 0 for i in range(n)],
				[1.0 if YY[j,j]==-1 else 0 for j in range(n)]]).T
	b = matrix([[1.0],[1.0]],(2,1))
	solvers.options['show_progress']=False
	sol = solvers.qp(P,p,G,h,A,b,initvals=init_sol)	
	margin2 = sol['primal objective']
	return margin2, sol['x'], sol
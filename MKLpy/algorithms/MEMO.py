# -*- coding: latin-1 -*-

"""
@author: Ivano Lauriola
@email: ivano.lauriola@phd.unipd.it

This file is part of MKLpy: a scikit-compliant framework for Multiple Kernel Learning
This file is distributed with the GNU General Public License v3 <http://www.gnu.org/licenses/>.  

"""

from .AlternateMKL import AlternateMKL
from ..arrange import average, summation
from ..lists.generator import HPK_generator
from ..metrics import frobenius
from sklearn.svm import SVC
import numpy as np
from cvxopt import spdiag,matrix,solvers


class MEMO(AlternateMKL):


	def __init__(self, learner=SVC(C=1000), generator=HPK_generator(n=10), multiclass_strategy='ova', verbose=False,
				theta=0.0, max_iter=1000, learning_rate=0.01, tolerance=1e-6, decay=0.8, callbacks=[]):
		super().__init__(learner=learner, generator=generator, multiclass_strategy=multiclass_strategy, func_form=summation,
				max_iter=max_iter, learning_rate=learning_rate, tolerance=tolerance, decay=decay, verbose=verbose, callbacks=callbacks)
		self.theta = theta
		self.direction = 'max'



	def initialize_optimization(self):
		Q = np.array([[np.dot(self.KL[r].ravel(),self.KL[s].ravel()) for r in range(self.n_kernels)] for s in range(self.n_kernels)])
		Q /= np.sum([frobenius(K)**2 for K in self.KL])
		#Q /= Q.mean()
		#Q /= len(self.Y)
		self.context = 	{
							'Q'		: Q,
							'YY'	: spdiag([1 if y==self.Y[0] else -1 for y in self.Y])
						}
		self.context['gamma'] = opt_margin(average(self.KL), self.context['YY'], None)[1]
		return np.ones(self.n_kernels)/self.n_kernels


	def do_step(self, lr, sol, w, obj):
		Kc = self.func_form(self.KL, w)
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
		_gamma = self.context['gamma']
		grad = np.array([(self.theta * np.dot(Q[r],w) + (_gamma.T * YY * matrix(self.KL[r]) * YY * _gamma)[0]) \
				* w[r] * (1 - w[r]) 	for r in range(self.n_kernels)])
		_beta = np.log(w) + lr * grad
		print (_beta)
		_w = np.exp(_beta)
		_w /= sum(_w)
		print (_beta,_w)
		_margin, _gamma, _sol = opt_margin(Kc, YY, sol)
		if _margin < 1e-4:
			print('margin')
			return None

		_comp = np.dot(_w,np.dot(_w,Q))
		_obj = _margin + (self.theta/2) * _comp
		self.context['gamma'] = _gamma
		return _obj, _w, _sol


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
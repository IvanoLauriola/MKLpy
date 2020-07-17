# -*- coding: latin-1 -*-

"""
@author: Ivano Lauriola
@email: ivano.lauriola@phd.unipd.it, ivanolauriola@gmail.com

This file is part of MKLpy: a scikit-compliant framework for Multiple Kernel Learning
This file is distributed with the GNU General Public License v3 <http://www.gnu.org/licenses/>.  

"""

import torch
from ..utils.validation import check_X

class Binarizer():
	''' base class for binarization algorithms '''
	def __init__(self, duplicate=False):
		self.duplicate = duplicate

	def fit(self,X,Y=None):
		raise NotImplementedError('Not implemented yet')

	def fit_transform(self,X,Y=None):
		self.fit(X,Y)
		return self.transform(X)

	def transform(self,X):
		raise NotImplementedError('Not implemented yet')

	def get_params(self,deep=True):
		return {'duplicate': self.duplicate}

	def set_params(self, **parameters):
		for parameter, value in parameters.items():
			setattr(self,parameter,value)
		return self


class AverageBinarizer(Binarizer):
	''' performs a feature discretization, 1 if the feature is over the average value threshold, 0 else '''

	def fit (self, X, Y=None):
		X = check_X(X)
		self.cols = torch.mean(X, dim=0)
		return self

	def transform(self, X, Y=None):
		X = check_X(X)
		if self.duplicate:
			Xb = torch.cat((X>self.cols,X<=self.cols),dim=-1)
		else:
			Xb = (X>self.cols)
		return Xb * 1

	def get_params(self, deep=True):
		return super().get_params()

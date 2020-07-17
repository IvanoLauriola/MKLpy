# -*- coding: latin-1 -*-

"""
@author: Ivano Lauriola
@email: ivano.lauriola@phd.unipd.it, ivanolauriola@gmail.com

This file is part of MKLpy: a scikit-compliant framework for Multiple Kernel Learning
This file is distributed with the GNU General Public License v3 <http://www.gnu.org/licenses/>.  

"""

class BinaryProblemError(ValueError):
	'''The target is not binary'''
	def __init__(self, n_classes, message=None):
		message = message or 'The target is not binary, %d classes found' % n_classes
		super().__init__(message)


class SquaredKernelError(ValueError):
	'''Kernel matrix is not squared'''
	def __init__(self,shape):
		message = 'K is not squared: %s' % str(shape)
		super().__init__(message)


class InvalidKernelsListError(TypeError):
	'''the kernel list is not valid'''
	def __init__(self):
		message = 'Required a kernels list'
		super().__init__(message)





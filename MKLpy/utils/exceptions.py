
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





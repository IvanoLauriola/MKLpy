
class BinaryProblemError(ValueError):
	'''The target is not binary'''
	def __init__(self,n_classes):
		message = 'The target is not binary, %d found' % n_classes
		super(SquaredKernelError, self).__init__(message)



class SquaredKernelError(ValueError):
	'''Kernel matrix is not squared'''
	def __init__(self,shape):
		message = 'K is not squared: %s' % str(shape)
		super(SquaredKernelError, self).__init__(message)


class InvalidKernelsListError(TypeError):
	'''the kernel list is not valid'''
	def __init__(self):
		message = 'KL is not a valid kernels list'
		super(InvalidKernelsListError, self).__init__(message)





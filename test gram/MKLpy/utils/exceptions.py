

class ArrangeMulticlassError(ValueError, AttributeError):
    '''asd'''
    

'''The kernel matrix is not squared'''
class SquaredKernelError(ValueError):
	def __init__(self,shape):
		message = 'K is not squared: %s' % str(shape)
		super(SquaredKernelError, self).__init__(message)


class InvalidKernelsListError(TypeError):
	def __init__(self):
		message = 'KL is not a valid kernels list'
		super(InvalidKernelsListError, self).__init__(message)





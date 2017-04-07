 
class MKL():

    how_to     = None   # the functional form in combination
    ker_matrix = None   # the obtained kernel matrix
    weights    = None   # the weights used in combination
    base       = None   # the base learner
    n_kernels  = None   # the number of kernels used in combination
    
    def arrange_kernel(self,X,Y):
        raise NotImplementedError
    

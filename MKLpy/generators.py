# -*- coding: latin-1 -*-

import types
from sklearn.metrics.pairwise import linear_kernel
from .metrics.pairwise import homogeneous_polynomial_kernel




class Lambda_generator:

    def __init__(self, X, Z=None, kernel=linear_kernel, params=None):
        self.X = X
        self.Z = Z#X if type(Z)==types.NoneType else Z
        self.kernel = kernel
        self.params = params
        self.n_kernels = len(params)

    def __len__(self):
        return self.n_kernels
    
    def __getitem__(self, r):
        #print (len(self.params), r)
        return self.kernel(self.X, self.Z, self.params[r])

    def __iter__(self):
        self.idx = 0
        #print ('restart')
        return self

    def __next__(self):
        if self.idx >= len(self):
            raise StopIteration
        #print (self.idx)
        self.idx += 1
        return self[self.idx-1]



class HPK_generator(Lambda_generator):

    def __init__(self, X, Z=None, degrees=range(1,11)):
        self.degrees = degrees
        self.params = range(1, degrees+1) if type(degrees) == int else degrees
        super().__init__(X=X, Z=Z, kernel=homogeneous_polynomial_kernel, params=degrees)



# -*- coding: latin-1 -*-

from .metrics import pairwise
import torch




class Generator:

    n_kernels = None

    def __init__(self, X, Z=None):
        self.X = X
        self.Z = X if (Z is None) or (Z is X) else Z

    def __len__(self):
        return self.n_kernels

    def __getitem__(self, r):
        raise NotImplementedError('This method has to be implemented in the derived class')

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx >= len(self):
            raise StopIteration
        self.idx += 1
        return self[self.idx-1]

    def to_list(self):
        return [K for K in self]



class Multiview_generator(Generator):
    # TODO
    pass



class Lambda_generator(Generator):

    def __init__(self, X, Z=None, kernels=[]):
        super().__init__(X=X, Z=Z)
        self.kernels   = kernels
        self.n_kernels = len(kernels)
    

    def __getitem__(self, r):
        return self.kernels[r](self.X, self.Z)



class HPK_generator(Generator):

    def __init__(self, X, Z=None, degrees=range(1,11), cache=True):
        super().__init__(X=X, Z=Z)
        self.degrees   = degrees
        self.cache     = cache
        self.n_kernels = len(degrees)
        if self.cache:
            self.L = pairwise.linear_kernel(self.X, self.Z)


    def __getitem__(self, r):
        L = self.L if self.cache else pairwise.linear_kernel(self.X, self.Z)
        return L**(r+1)



class RBF_generator(Generator):

    def __init__(self, X, Z=None, gamma=[0.01, 0.1, 1], cache=True):
        super().__init__(X=X, Z=Z)
        self.gamma   = gamma
        self.cache     = cache
        self.n_kernels = len(gamma)
        if self.cache:
            self.D = pairwise.euclidean_distances(self.X, self.Z)**2


    def __getitem__(self, r):
        D = self.D if self.cache else pairwise .euclidean_distances(self.X, self.Z)**2
        return torch.exp(-self.gamma[r] * D)




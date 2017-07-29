# -*- coding: latin-1 -*-
import random
import numpy as np
from kernel_list import Kernel_list as klist
from sklearn.metrics.pairwise import linear_kernel
import types

class Generator(object):
    p = 1

    def __init__(self, n=10, typ='function',base=linear_kernel):
        self.typ = typ
        self.n = n
        self.base = base

    def _next(self, X, T=None):
        raise NotImplementedError('not implemented yet')

    def make_a_list(self, X, T=None):
        self.L = self.base(X,T) if self.base else None
        T = X if type(T)==types.NoneType else T
        KL = klist()
        for i in range(self.n):
            KL.append(self._next(X,T))
        return KL


class HPK_generator(Generator):

    def __init__(self, n=10, typ='function'):
        super(self.__class__, self).__init__(n=n, typ=typ)

    def _next(self, X, T):
        p = self.p
        if self.typ=='storage':
            K = self.L ** p
            pass    # TODO
        elif self.typ=='matrix':
            K = self.L ** p
            kernel = lambda : K
        elif self.typ=='function' or True:
            kernel = lambda : self.L**p
        self.p += 1
        return kernel

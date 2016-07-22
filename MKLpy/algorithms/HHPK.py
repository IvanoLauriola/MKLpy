# -*-coding: latin-1 -*-

from base import MKL
from MKLpy.arrange import summation
from MKLpy.regularization import tracenorm
from MKLpy.lists import SFK_generator, HPK_generator
from MKLpy.utils.validation import check_kernel_list
import numpy as np
import math
from sklearn.metrics.pairwise import linear_kernel,polynomial_kernel

class HHPK(MKL):

    def __init__(self, h='linear', m=1,  n_kernels=10, verbose=False):
        self.h = h
        self.n_kernels = n_kernels
        self.verbose = verbose
        self.m = m
        
    def _heuristics(self):
        if self.h == 'fixed_log':
            return self.fixed_log
        if self.h == 'exp':
            return self.base
        if self.h == 'log':
            return self.log
        elif self.h == 'linear' or True:
            return lambda k_list : summation(k_list,[i*self.m +1 for i in range(self.n_kernels)])
    
    def arrange_kernel(self, X, X_te=None, Y=None, check=True):
        
        f = self._heuristics()
        K = f(X,X_te)
        return K

    def fixed_log(self,X,X_te):
        k = linear_kernel(X_te,X)
        K = np.zeros(k.shape)
        for i in range(self.n_kernels):
            K += (k**(i+1)) * math.log(i+1)
        self.weights = [math.log(i+1) for i in range(self.n_kernels)]
        return K

    def base(self,X,X_te):
        w = [1]
        K = np.zeros((X_te.shape[0],X.shape[0]), dtype = np.double)
        for i in range(self.n_kernels):
            w.append(w[i]*self.m)
        norm = sum([ww for ww in w])
        w = [ww /norm for ww in w]
        self.weights = w
        for i in range(self.n_kernels):
            K += polynomial_kernel(X_te,X,degree=i+1,gamma=1,coef0=0) * w[i]
        return K
    
    def log(self,X,X_te):
        return








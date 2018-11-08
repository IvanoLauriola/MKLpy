# -*- coding: latin-1 -*-
import numpy as np
import types

class Kernel_list():
    def __init__(self, KL=[]):
        self.KL = []
        for K in KL:
            self.append(K)
    
    def __getitem__(self,i):
        if type(i) != int:
            raise TypeError('Index must be int')
        if i >= len(self.KL):
            raise IndexError('This kernel is not generated yet')
        return self.KL[i]()
        # KL[i] is always a function
            
    def __len__(self):
        '''returns the length of kernels list'''
        return len(self.KL)
    	
    def __str__(self):
        return 'TODO'
    
    def append(self, kernel):
        '''appends a kernel to list'''
        #if type(kernel) == str:                 # path
        #    f = lambda : np.load(kernel)
        #else : if type(kernel) == np.ndarray:   # kernel matrix
        #    f = lambda : kernel
        #else : if hasattr(kernel, '__call__'):  # kernel function
        #    f = kernel
        #else : raise TypeError('kernel is not a kernel, %s' % str(type(kernel)))
        #self.KL.append(f)
        self.KL.append(kernel)
    
    def to_array(self):
        '''returns a list with all computed kernels'''
        return [f() for f in self.KL]

    def compute(i):
        '''compute a kernel and keep it in memory'''
        self.KL[i] = lambda : self.KL[i]()
    
    def edit(self,f):
        '''apply the function f to all kernels'''
        self.KL = [lambda : f(g()) for g in self.KL]

    def extend(self,KL):
        for K in KL:
            self.append(K)

    def __add__(self,KL):
        self.extend(KL)
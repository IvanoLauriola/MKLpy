# -*- coding: latin-1 -*-
from cvxopt import matrix
import numpy as np
from sklearn.metrics.pairwise import linear_kernel
import types

class kernel_list():
    def __init__(self, X, T=None, k_list=None):
        self.X = X
        T = X if type(T)==types.NoneType else T
        self.T = T
        self.func_list = []
        self.feat_list = []
        self.shape = (0,T.shape[0],X.shape[0])
        self.transform = lambda X : X
        
        if k_list:
            self.func_list = [f for f in k_list.func_list]
            self.feat_list = [f for f in k_list.feat_list]
            self.shape = k_list.shape

    
    def __getitem__(self,i, key='gram'):
        if type(i) != int:
            raise TypeError('Index must be int')
        if i >= self.shape[0]:
            raise IndexError('This kernel is not generated yet')
        if key == 'func':
            return self.func_list[i]
        elif key == 'feature':
            return self.feat_list[i]
        elif key == 'gram' or True:
            features = self.feat_list[i]
            return self.transform(   self.func_list[i](self.T[:,features], self.X[:,features])   )
            
    def __len__(self):
        return self.shape[0]
    	
    def __str__(self):
        header = "num" + "\tfeature\n"
        value ='\n'.join( "" + (i+1).__str__() +"\t"+ self.feat_list[i].__str__() for i in range(self.shape[0]) ) 
        return header + value
    
    def append(self, kernel, feature_list = [], params = {}):#, c=1):
        if not hasattr(kernel, '__call__'):
            raise TypeError('kernel function must be callable')
        f = lambda X,Y : kernel(X, Y, **params) #fisso i parametri altrimenti li vado perdendo
        self.func_list.append(f)
        self.feat_list.append(feature_list)
        self.shape = (self.shape[0]+1, self.shape[1], self.shape[2])
    
    def __add__(self, o):
        l = kernel_list(self.X,self.T,self)
        for i in range(len(o)):
            l.append(o.__getitem__(i,'func'),feature_list = o.__getitem__(i,'feature'))
        return l
    def __iadd__(self,o):
        for i in range(len(o)):
            self.append(o.__getitem__(i,'func'),feature_list = o.__getitem__(i,'feature'))
        return self
    
    def __div__(self, o):
        c = self._check_op(o)
        l = kernel_list(self.X,self.T,self)
        old = l.func_list
        l.func_list = [lambda X,T,i=i : old[i](X,T)/c[i] for i in range(self.shape[0])]
        return l

    '''
    def __floordiv__(self, o):
        return self.__div__(o)
    def __truediv__(self, o):
        return self.__div__(o)
    '''
    def __mul__(self,o):
        c = self._check_op(o)
        return self.__div__(list(1.0/np.array(c)))
    
    def _check_op(self,o):
        return o if hasattr(o,'__len__') else np.full(self.shape[0],o)
    
    def __idiv__(self,o):
        c = self._check_op(o)
        old = self.func_list
        self.func_list = [lambda X,T,i=i : old[i](X,T)/c[i] for i in range(self.shape[0])]
        return self

    def __imul__(self,o):
        c = self._check_op(o)
        return self.__idiv__(list(1.0/np.array(c)))
    
    def to_array(self):
        return np.array([self[i] for i in range(self.shape[0])])
    
    def edit(self,f):
        oldt = self.transform
        self.transform = lambda X : f(oldt(X))


import random
import numpy as np
#from kernel_list import kernel_list as klist
from kernel_list import kernel_list as klist
from sklearn.metrics.pairwise import linear_kernel, polynomial_kernel, rbf_kernel
import types

class generator():
    def next(self):
        raise NotImplementedError('not implemented yet')

    def make_a_list(self,n):
        k_list = klist(self.X,self.T)
        for i in range(n):
            k_list.append(*self.next())
        return k_list


class weak_generator(generator):
    def __init__(self, X, T=None, func='rbf', params={}, n_feature=2):
        self.X = X
        self.T = T
        self.n_feature = n_feature
        self.f_max = len(X[0])
        
        if hasattr(func, '__call__'):
            self.func = lambda X,Y : func(X,Y, **params)
        elif func == 'linear':
            self.func = lambda X,Y : linear_kernel(X,Y, **params)
        elif func == 'poly':
            self.func = lambda X,Y : polynomial_kernel(X,Y, **params)
        elif func == 'rbf' or True:
            self.func = lambda X,Y : rbf_kernel(X,Y, **params)


class sequential_generator(generator):
    def __init__(self,X, T=None):
        self.X = X
        self.T = T
        self.p = 1
        T = X if type(T)==types.NoneType else T
        self.base = linear_kernel(T,X)    

class SFK_generator(weak_generator):
    '''Sample Feature Kernels'''
    def next(self):
        random_state = np.random.RandomState()
        feature_list = [random.choice(range(0,self.f_max)) for i in range(self.n_feature)]
        return self.func,feature_list

class HPK_generator(sequential_generator):
    '''Homogeneous Polynomial Kernels'''    
    def next(self):
        p = self.p
        #f = lambda X,Y: polynomial_kernel(X,Y,degree=p,gamma=1.0,coef0=0.0)
        f = lambda X,Y : self.base**p
        self.p += 1
        return f, range(len(self.X[0])) 

class SSK_generator(sequential_generator):
    #TODO
    def next(self):
        raise NotImplementedError('not implemented yet')




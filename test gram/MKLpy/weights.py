import numpy as np
from cvxopt import matrix,solver


#making weights vectors

def uniform_vector(n):
    return np.ones(n)/n





#--------------------------------------






# some post-process function

class transformation():

    def __init__(self):
        pass
    
    def transform(self, w):
        raise NotImplementedError


class base_transformation(transformation):
    def __init__(self):
        pass
    
    def transform(self, w):
        '''returns a weights vector w where \|w\|_1 = 1'''
        return np.array(w)/np.linalg.norm(w,1)


class std_transformation(transformation):
    def __init__(self,lam):
        self.lam = lam
    
    def transform(self,w):
        '''w = argmin lam||w||^2_2 + (1-lam)(-old_w)Tw'''
        #TODO check the optimization problem
        
        n = len(w)
        P = spdiag([self.lam*2.0]*n) * self.lam
        q = -matrix(w) * (1 - self.lam)
        A = matrix([[1.0]*n]).T
        b = matrix([1.0], (1,1))
        #positive constraints
        G = -spdiag([-1.0]*n)
        h = matrix([0.0]*n, (n,1))
        if self.non_negative:
            sol = solver.qp(P,q,G,h,A,b,initvals = matrix(w))
        else:
            sol = solver.qp(P,q,A=A,b=b,initvals = matrix(w))
        return [_w for _w in sol['x']]


class top_transformation(transformation):
    def __init__(self,top):
        self.top = top
    
    def transform(self,w):
        '''returns the top kernels, if top is a float value [0,1] then
        returns the percentage of top kernels
        else, top is an integer value, then returns the top kernels'''
        if self.top <= 0:
            raise ValueError('Top must be greater than 0')
        n = len(w)
        top = self.top if type(self.top)==int else int(n * self.top)
        
        # TODO
        pass

"""
@author: Michele Donini
@email: mdonini@math.unipd.it

EasyMKL: a scalable multiple kernel learning algorithm
by Fabio Aiolli and Michele Donini

Paper @ http://www.math.unipd.it/~mdonini/publications.html
"""

from cvxopt import matrix, solvers, mul
import numpy as np


class EasyMKL():
    ''' EasyMKL is a Multiple Kernel Learning algorithm.
        The parameter lam (lambda) has to be validated from 0 to 1.

        For more information:
        EasyMKL: a scalable multiple kernel learning algorithm
            by Fabio Aiolli and Michele Donini

        Paper @ http://www.math.unipd.it/~mdonini/publications.html
    '''
    def __init__(self, lam = 0.1, tracenorm = True):
        self.lam = lam
        self.tracenorm = tracenorm
        
        self.list_Ktr = None
        self.labels = None
        self.gamma = None
        self.weights = None
        self.traces = []

    def sum_kernels(self, list_K, weights = None):
        ''' Returns the kernel created by averaging of all the kernels '''
        k = matrix(0.0,(list_K[0].size[0],list_K[0].size[1]))
        if weights == None:
            for ker in list_K:
                k += ker
        else:
            for w,ker in zip(weights,list_K):
                k += w * ker            
        return k
    
    def traceN(self, k):
        return sum([k[i,i] for i in range(k.size[0])]) / k.size[0]
    
    def train(self, list_Ktr, labels):
        ''' 
            list_Ktr : list of kernels of the training examples
            labels : array of the labels of the training examples
        '''
        self.list_Ktr = list_Ktr  
        for k in self.list_Ktr:
            self.traces.append(self.traceN(k))
        #return self.traces
        #print self.traces
        
        if self.tracenorm:
            #self.list_Ktr = [k / self.traceN(k) for k in list_Ktr]
            self.list_Ktr = [k/self.traces[i] for i,k in enumerate(list_Ktr)]
        #return self.list_Ktr[0]

        set_labels = set(labels)
        if len(set_labels) != 2:
            print 'The different labels are not 2'
            return None
        elif (-1 in set_labels and 1 in set_labels):
            self.labels = labels
        else:
            poslab = max(set_labels)
            self.labels = matrix(np.array([1. if i==poslab else -1. for i in labels]))
        
        # Sum of the kernels
        ker_matrix = matrix(self.sum_kernels(self.list_Ktr))

        YY = matrix(np.diag(list(matrix(self.labels))))
        KLL = (1.0-self.lam)*YY*ker_matrix*YY
        LID = matrix(np.diag([self.lam]*len(self.labels)))
        Q = 2*(KLL+LID)
        p = matrix([0.0]*len(self.labels))
        G = -matrix(np.diag([1.0]*len(self.labels)))
        h = matrix([0.0]*len(self.labels),(len(self.labels),1))
        A = matrix([[1.0 if lab==+1 else 0 for lab in self.labels],[1.0 if lab2==-1 else 0 for lab2 in self.labels]]).T
        b = matrix([[1.0],[1.0]],(2,1))
        
        solvers.options['show_progress']=False#True
        sol = solvers.qp(Q,p,G,h,A,b)
        # Gamma:
        self.gamma = sol['x']   
        self.g1 = [x for x in self.gamma]  
        
        # Bias for classification:
        bias = 0.5 * self.gamma.T * ker_matrix * YY * self.gamma
        self.bias = bias
        
        # Weights evaluation:
        yg =  mul(self.gamma.T,self.labels.T)
        self.weights = []
        for kermat in self.list_Ktr:
            b = yg*kermat*yg.T
            self.weights.append(b[0])
        
        norm2 = sum([w for w in self.weights])
        self.weights = [w / norm2 for w in self.weights]

        if self.tracenorm: 
            for idx,val in enumerate(self.traces):
                self.weights[idx] = self.weights[idx] / val        
        
        if True:
            ker_matrix = matrix(self.sum_kernels(self.list_Ktr, self.weights))
            self.ker_matrix = ker_matrix
        
            YY = matrix(np.diag(list(matrix(self.labels))))
            
            KLL = (1.0-self.lam)*YY*ker_matrix*YY
            LID = matrix(np.diag([self.lam]*len(self.labels)))
            Q = 2*(KLL+LID)
            p = matrix([0.0]*len(self.labels))
            G = -matrix(np.diag([1.0]*len(self.labels)))
            h = matrix([0.0]*len(self.labels),(len(self.labels),1))
            A = matrix([[1.0 if lab==+1 else 0 for lab in self.labels],[1.0 if lab2==-1 else 0 for lab2 in self.labels]]).T
            b = matrix([[1.0],[1.0]],(2,1))
            
            solvers.options['show_progress']=False#True
            sol = solvers.qp(Q,p,G,h,A,b)
            # Gamma:
            self.gamma = sol['x']
        
        return self
    
    def rank(self,list_Ktest):
        '''
            list_Ktr : list of kernels of the training examples
            labels : array of the labels of the training examples
            Returns the list of the examples in test set of the kernel K ranked
        '''
        if self.weights == None:
            print 'EasyMKL has to be trained first!'
            return
         
        #YY = matrix(np.diag(self.labels).copy())
        YY = matrix(np.diag(list(matrix(self.labels))))
        ker_matrix = matrix(self.sum_kernels(list_Ktest, self.weights))
        z = ker_matrix*YY*self.gamma
        return z

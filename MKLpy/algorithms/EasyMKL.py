# -*- coding: latin-1 -*-
"""
@author: Michele Donini
@email: mdonini@math.unipd.it
 
EasyMKL: a scalable multiple kernel learning algorithm
by Fabio Aiolli and Michele Donini
 
Paper @ http://www.math.unipd.it/~mdonini/publications.html
"""
from sklearn.base import BaseEstimator, ClassifierMixin
from base import MKL
from sklearn.utils import check_array, check_consistent_length#, check_random_state
from sklearn.utils import column_or_1d, check_X_y
from sklearn.utils import validation
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import NotFittedError
from sklearn.utils.multiclass import check_classification_targets
from MKLpy.multiclass import OneVsOneMKLClassifier as ovoMKL, OneVsOneMKLClassifier as ovaMKL   #ATTENZIONE DUE OVO
from MKLpy.utils.exceptions import ArrangeMulticlassError
from MKLpy.algorithms import KOMD
 
from cvxopt import matrix, spdiag, solvers
import numpy as np
 
from MKLpy.arrange import summation
from MKLpy.regularization import tracenorm
from MKLpy.lists import HPK_generator
from MKLpy.utils.validation import check_KL_Y
 
 
class EasyMKL(BaseEstimator, ClassifierMixin, MKL):
    ''' EasyMKL is a Multiple Kernel Learning algorithm.
        The parameter lam (lambda) has to be validated from 0 to 1.
 
        For more information:
        EasyMKL: a scalable multiple kernel learning algorithm
            by Fabio Aiolli and Michele Donini
 
        Paper @ http://www.math.unipd.it/~mdonini/publications.html
    '''
    def __init__(self, lam = 0.1, base = None, kernel='precomputed', tracenorm = False, gen = None, n_kernels = 10, max_iter = 100, verbose = False, multiclass_strategy = 'ova'):
        self.lam = lam
        self.kernel = kernel
        self.gen = gen if gen else HPK_generator #gen dev'essere la classe, non l'istanza
        self.n_kernels = n_kernels
        self.max_iter = max_iter
        self.verbose = verbose
        self.multiclass_strategy = multiclass_strategy
        self.tracenorm = tracenorm
        
        self.KL = None
        self.Y = None
        self.gamma = None
        #self.traces = []
        self.is_fitted = False
        self.multiclass_ = None
        self.classes_ = None
        
        self.how_to = summation
        self.weights = None
        self.base = base if base else KOMD(lam=lam,kernel='precomputed',verbose=verbose)
        self.base.kernel='precomputed'
        
    def fit(self, X, Y):
        self._input_checks(X,Y)
        self.multiclass = len(self.classes_) > 2
        if not self.multiclass:
            return self._fit()
        else :
            fm = ovoMKL if self.multiclass_strategy == 'ovo' else ovaMKL
            self.cls = fm(EasyMKL(**self.get_params()),self.base).fit(X,Y)
            self.is_fitted = True
            return self
        raise ValueError('This is a very bad exception...')
     
    def _fit(self): #qua sicuro è binario      #madonna se è binario
        ker_matrix = matrix(self.arrange_kernel(self.KL,self.Y))
        self.ker_matrix = ker_matrix
        self.base = self.base.fit(ker_matrix,self.Y)
        self.is_fitted = True
        return self
        
    def arrange_kernel(self,X,Y):
        if len(np.unique(Y)) > 2:
            raise ArrangeMulticlassError('arrange_kernel does not work in multiclass context')
        self._input_checks(X,Y)

        self.Y = [1 if y==self.classes_[1] else -1 for y in self.Y]
        n_sample = len(self.Y)
        ker_matrix = matrix(summation(self.KL))
        YY = spdiag(self.Y)
        KLL = (1.0-self.lam)*YY*ker_matrix*YY
        LID = spdiag([self.lam]*n_sample)
        Q = 2*(KLL+LID)
        p = matrix([0.0]*n_sample)
        G = -spdiag([1.0]*n_sample)
        h = matrix([0.0]*n_sample,(n_sample,1))
        A = matrix([[1.0 if lab==+1 else 0 for lab in self.Y],[1.0 if lab2==-1 else 0 for lab2 in self.Y]]).T
        b = matrix([[1.0],[1.0]],(2,1))
         
        solvers.options['show_progress'] = False
        solvers.options['maxiters'] = self.max_iter
        sol = solvers.qp(Q,p,G,h,A,b)
        self.gamma = sol['x']
        if self.verbose:
            print '[EasyMKL] - first step'
            print 'optimization finished, #iter = ', sol['iterations']
            print 'status of the solution: ', sol['status']
            print 'objval: ', sol['primal objective']
         
        # Bias for classification:
        #bias = 0.5 * self.gamma.T * ker_matrix * YY * self.gamma
        #self.bias = bias
        
        # Weights evaluation:
        #yg =  mul(self.gamma.T,matrix(self.Y).T)
        yg = self.gamma.T * YY
        self.weights = []
        for kermat in self.KL:
            b = yg*matrix(kermat)*yg.T
            self.weights.append(b[0])
         
        norm2 = sum([w for w in self.weights])
        self.weights = [w / norm2 for w in self.weights]
        ker_matrix = summation(self.KL, self.weights)
        self.ker_matrix = ker_matrix
        return ker_matrix
    
     
     
    def predict(self, X):
        KL = self._check_test(X)
        return self.cls.decision_function(KL) if self.multiclass_ else self.base.predict(summation(KL, self.weights))
         
    def decision_function(self, X):
        KL = self._check_test(X)
        return self.cls.decision_function(KL) if self.multiclass_ else self.base.decision_function(summation(KL, self.weights))
 
    def get_params(self, deep=True):
        # this estimator has parameters:
        return {"lam": self.lam, "tracenorm": self.tracenorm, "kernel": self.kernel, 
                "gen": self.gen, "n_kernels": self.n_kernels, "max_iter":self.max_iter,
                "verbose":self.verbose, "multiclass_strategy":self.multiclass_strategy}
 
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self,parameter,value)
        return self
    
    def _input_checks(self, X, Y):
        '''valida l'input dell'algoritmo e fa un po di preprocessing'''
        if self.kernel == 'precomputed':
            self.KL = check_KL_Y(X,Y)
        else:
            X,Y = validation.check_X_y(X, Y, dtype=np.float64, order='C', accept_sparse='csr')
            self.KL = self.gen(X).make_a_list(self.n_kernels)
        check_classification_targets(Y)
        self.Y = Y
        self.X = X
        self.classes_ = np.unique(Y)
        if len(self.classes_) < 2:
            raise ValueError("The number of classes has to be almost 2; got ", len(self.classes_))
        
        if self.tracenorm:
            for k in self.KL:
                self.traces.append(traceN(k))

    def _check_test(self,X):
        if self.is_fitted == False:
            raise NotFittedError("This EasyMKL instance is not fitted yet. Call 'fit' with appropriate arguments before using this method.")
        if self.kernel == 'precomputed':
            KL = check_KL_Y(X,self.Y)
        else:
            X = check_array(X, accept_sparse='csr', dtype=np.float64, order="C")
            if X.shape[1] != self.X.shape[1]:#self.n_f:
                raise ValueError("The number of feature in X not correspond")
            KL = self.gen(self.X,X).make_a_list(self.n_kernels).to_array()
        return KL



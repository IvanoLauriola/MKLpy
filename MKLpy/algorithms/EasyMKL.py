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
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
 
from cvxopt import matrix, solvers, mul
import numpy as np
import random
 
from MKLpy.arrange import summation
from MKLpy.regularization import tracenorm
from MKLpy.lists import SFK_generator, HPK_generator
from MKLpy.utils.validation import check_kernel_list
 
class EasyMKL(BaseEstimator, ClassifierMixin, MKL):
    ''' EasyMKL is a Multiple Kernel Learning algorithm.
        The parameter lam (lambda) has to be validated from 0 to 1.
 
        For more information:
        EasyMKL: a scalable multiple kernel learning algorithm
            by Fabio Aiolli and Michele Donini
 
        Paper @ http://www.math.unipd.it/~mdonini/publications.html
    '''
    def __init__(self, lam = 0.1, tracenorm = True, kernel = 'rbf', gen = None, n_kernels = 50, rbf_gamma = 0.1, degree = 2.0, coef0 = 0.0, max_iter = 100, verbose = False, multiclass_strategy = 'ova'):
        self.lam = lam
        self.tracenorm = tracenorm
        self.kernel = kernel
        self.gen = gen
        self.n_kernels = n_kernels
        self.rbf_gamma = rbf_gamma
        self.degree = degree
        self.coef0 = coef0
        self.max_iter = max_iter
        self.verbose = verbose
        self.multiclass_strategy = multiclass_strategy
         
        self.K = None
        self.Y = None
        self.gamma = None
        self.weights = None
        self.traces = []
        self.is_fitted = False
        self.multiclass_ = None
        self.classes_ = None
     
         
    def arrange_kernel(self,X,Y,check=True):
        if check:
            self._input_checks(X,Y)
         
        #if len(self.classes_) != 2:
        #    raise ValueError("The number of classes has to be 2; got ", len(self.classes_))
         
        self.Y = [1 if y==self.classes_[1] else -1 for y in self.Y]
        n_sample = len(self.Y)
        ker_matrix = matrix(summation(self.K))
        #ker_matrix = ker_matrix / len(X)
        YY = matrix(np.diag(list(matrix(self.Y))))
        KLL = (1.0-self.lam)*YY*ker_matrix*YY
        LID = matrix(np.diag([self.lam]*n_sample))
        Q = 2*(KLL+LID)
        p = matrix([0.0]*n_sample)
        G = -matrix(np.diag([1.0]*n_sample))
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
        bias = 0.5 * self.gamma.T * ker_matrix * YY * self.gamma
        self.bias = bias
         
        # Weights evaluation:
        #yg =  mul(self.gamma.T,matrix(self.Y).T)
        yg = self.gamma.T * YY
        self.weights = []
        for kermat in self.K:
            b = yg*matrix(kermat)*yg.T
            self.weights.append(b[0])
         
        norm2 = sum([w for w in self.weights])
        #norm2 = sum(self.weights)
        self.weights = [w / norm2 for w in self.weights]
         
        if self.tracenorm: 
            for idx,val in enumerate(self.traces):
                self.weights[idx] = self.weights[idx] / val        
        
        ker_matrix = summation(self.K, self.weights)
        self.ker_matrix = ker_matrix
        #return matrix(summation(self.K, self.weights))
        return ker_matrix
 
    def how_to(self):
        return summation
         
         
    def fit(self, X, Y):
        self._input_checks(X,Y)
         
        #il multiclass è da sistemare
        if len(self.classes_) == 2:
             
            self.multiclass_ = False
            return self._fit()
        else :
            self.multiclass_ = True
            if self.multiclass_strategy == 'ovo':
                return self._one_vs_one(self.X,self.Y)
            else :
                return self._one_vs_rest(self.X,self.Y)
        raise ValueError('This is a very bad exception...')
     
     
    def _fit(self):
        ker_matrix = matrix(self.arrange_kernel(self.X,self.Y,check=False))
        self.ker_matrix = ker_matrix
        n_sample = len(self.Y)
        YY = matrix(np.diag(list(matrix(self.Y))))
        KLL = (1.0-self.lam)*YY*ker_matrix*YY
        LID = matrix(np.diag([self.lam]*n_sample))
        Q = 2*(KLL+LID)
        #GIA CALCOLATE (ma dividendo devo ricalcolarle..........)
        p = matrix([0.0]*n_sample)
        G = -matrix(np.diag([1.0]*n_sample))
        h = matrix([0.0]*n_sample,(n_sample,1))
        A = matrix([[1.0 if lab==+1 else 0 for lab in self.Y],[1.0 if lab2==-1 else 0 for lab2 in self.Y]]).T
        b = matrix([[1.0],[1.0]],(2,1))
        solvers.options['show_progress'] = False
        solvers.options['maxiters'] = self.max_iter
        sol = solvers.qp(Q,p,G,h,A,b)
        self.gamma = sol['x']
        if self.verbose:
            print '[EasyMKL] - second step'
            print 'optimization finished, #iter = ', sol['iterations']
            print 'status of the solution: ', sol['status']
            print 'objval: ', sol['primal objective']
        self.is_fitted = True
        bias = 0.5 * self.gamma.T * ker_matrix * YY * self.gamma
        self.bias = bias
        return self
     
    def wut_predict_proba(self, X):
        if self.multiclass_:
            return self.cls.predict_proba(X)
        v = self.decision_function(X)
        b = -v
        return np.array([v,b]).T
     
    def predict(self, X):
        if self.is_fitted == False:
            raise NotFittedError("This EasyMKL instance is not fitted yet. Call 'fit' with appropriate arguments before using this method.")
        if self.multiclass_ == True:
            return self.cls.predict(X)
        return np.array([self.classes_[1] if p >= 0 else self.classes_[0] for p in self.decision_function(X)])
         
    def decision_function(self, X):
        if self.is_fitted == False:
            raise NotFittedError("This EasyMKL instance is not fitted yet. Call 'fit' with appropriate arguments before using this method.")
        
        if self.kernel == 'precomputed':
            K = check_kernel_list(X)
        else:
            X = check_array(X, accept_sparse='csr', dtype=np.float64, order="C")
            if X.shape[1] != self.n_f:
                raise ValueError("The number of feature in X not correspond")
            #K = self.K.set_test(X)
            K = kernel_list(self.X,X,self.K)

        if self.multiclass_ == True:
            return self.cls.decision_function(X)
         
        YY = matrix(np.diag(list(matrix(self.Y))))
        ker_matrix = matrix(summation(K, self.weights))
        z = ker_matrix*YY*self.gamma
        #z = z-self.bias
        return np.array(list(z))
 
 
 
     
    def get_params(self, deep=True):
        # this estimator has parameters:
        return {"lam": self.lam, "tracenorm": self.tracenorm, "kernel": self.kernel, 
                "gen": self.gen, "n_kernels": self.n_kernels, "rbf_gamma":self.rbf_gamma,
                "degree":self.degree, "coef0":self.coef0, "max_iter":self.max_iter,
                "verbose":self.verbose, "multiclass_strategy":self.multiclass_strategy}
 
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self,parameter,value)
        return self
     
    def _input_checks(self, X, Y):
        '''valida l'input dell'algoritmo e fa un po di preprocessing'''
        if self.kernel == 'precomputed':
            self.K = check_kernel_list(X)
        elif self.gen != None:
            X,Y = validation.check_X_y(X, Y, dtype=np.float64, order='C', accept_sparse='csr')
            self.K = self.gen.make_a_list(self.n_kernels)
        else:
            random.seed(1)
            X,Y = validation.check_X_y(X, Y, dtype=np.float64, order='C', accept_sparse='csr')
            self.K = random_generator(X).make_a_list(self.n_kernels)#.to_array()    #da sistemare i parametri
            #self.K = poly_generator(X).make_a_list(20)#.to_array()
        check_classification_targets(Y)
        self.Y = Y
        self.X = X
        self.classes_ = np.unique(Y)
        if len(self.classes_) < 2:
            raise ValueError("The number of classes has to be almost 2; got ", len(self.classes_))
        self.traces = []
        #for k in self.K:
        #    self.traces.append(traceN(k))
        if self.tracenorm and False:
            for k in self.K:
                self.traces.append(traceN(k))
            from MKL.utils import kernel_list
            if not type(self.K).__name__ =='ndarray':#== kernel_list:
                #print 'k_list'
                self.K /= self.traces
            else:
                #print 'array'
                c = [   [[i for j in range(len(Y))] for k in range(len(Y))] for i in self.traces   ]
                self.K /= np.array(c)
                del c
        self.n_f = X.shape[1]
 
    #il meta-estimatore non lavora con liste di kernel, quindi se il kernel è precomputed
    # sono fottuto perché non posso passare K (salta su check_X_y)
    # mi sa che devo implementarlo a mano
    # TIP: al meta estimatore devo passare per forza la lista dei kernel, 
    #      perché se se la genera da solo utilizzerà sempre kernel diversi.
    #      Attualmente però gli passo X e si ricalcola kernel diversi,
    #      ok che con un numero alto si converge allo stesso risultato
    #      ma comunque dubito sia corretto
    def _one_vs_one(self,X,Y):
        self.cls = OneVsOneClassifier(EasyMKL(**self.get_params())).fit(X,Y)
        self.is_fitted = True
        return self
     
    def _one_vs_rest(self,X,Y):
        self.cls = OneVsRestClassifier(EasyMKL(**self.get_params())).fit(X,Y)   #self.K
        self.is_fitted = True
        return self

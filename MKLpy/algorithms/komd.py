"""
.. codeauthor:: Michele Donini <m.donini@ucl.ac.uk>
.. codeauthor:: Ivano Lauriola <ivanolauriola@gmail.com>

A Kernel Method for the Optimization of the Margin Distribution
"""
 
from cvxopt import matrix, solvers
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel, linear_kernel
from sklearn.utils import check_array, check_consistent_length#, check_random_state
from sklearn.utils import column_or_1d, check_X_y
from sklearn.utils import validation
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import NotFittedError
from sklearn.utils.multiclass import check_classification_targets
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier

class KOMD(BaseEstimator, ClassifierMixin):
    """KOMD.
    
    KOMD is a kernel method for classification and ranking.
    
    Read more in http://www.math.unipd.it/~dasan/papers/km-omd.icann08.pdf
    by F. Aiolli, G. Da San Martino, and A. Sperduti.
    
    For details on the precise mathematical formulation of the provided
    kernel functions and how `gamma`, `coef0` and `degree` affect each
    other, see the corresponding section in the narrative documentation:
    :ref:`svm_kernels`.
	
    Parameters
    ----------
    lam : float, (default=0.1)
        Specifies the lambda value, between 0.0 and 1.0.
    
    kernel : optional (default='linear')
        Specifies the kernel function used by the algorithm.
        It must be one of 'linear', 'poly', 'rbf', a callable or a gram matrix.
        If none is given, 'linear' will be used. If a callable is given it is
        used to pre-compute the kernel matrix from data matrices; that matrix
        should be an array of shape ``(n_samples, n_samples)``.
    
    rbf_gamma : float, optional (default=0.1)
        Coefficient for 'rbf' and 'poly' kernels.
        Ignored by all other kernels.
    
    degree : float, optional (default=2.0)
        Specifies the degree of the 'poly' kernel.
	    Ignored by all other kernels.
    
    coef0 : flaot, optional (default=0.0)
        Specifies the coeff0 in a polynomial kernel.
        Ignored by all other kernels.
    
    max_iter : int, optional (default=100)
        Hard limit on iterations within solver, it can't be negative.
    
    verbose : bool, (default=False)
        Enable verbose output during fit.
    
    multiclass_strategy : string, optional (default='ova')
        Specifies the strategy used in case of multiclass.
        'ova' for one_vs_all pattern (also called one_vs_rest),
        'ovo' for one_vs_one pattern.
        With other unexpected string, 'ova' pattern is used.
    
    Attributes
    ----------
    gamma : array-like, shape = [n_samples]
        probability-like vector that define the distance vector
        over the two class.
    
    classes_ : array-like, shape = [n_classes]
        Vector that contain all possibile labels
    
    multiclass_ : boolean,
        True if the number of classes > 2
    
    Examples
    --------
    >>>import numpy as np
    >>>from ??.komd import KOMD
    >>>X = np.array([[1,2,i] for i in range(5)])
    >>>Y = np.array([1,1,1,-1,-1])
    >>>cls = KOMD()
    >>>cls = cls.fit(X,Y)
    >>>pred = cls.predict([[1,1,5]])
    
    References
    ----------
    `A Kernel Method for the Optimization of the Margin Distribution
    <http://www.math.unipd.it/~dasan/papers/km-omd.icann08.pdf>`__
    """
    
    def __init__(self, lam = 0.1, kernel = 'rbf', rbf_gamma = 0.1, degree = 2.0, coef0 = 0.0, max_iter = 100, verbose = False, multiclass_strategy = 'ova'):
        self.lam = lam
        self.gamma = None
        self.bias = None
        self.X = None
        self.Y = None
        self.is_fitted = False
        self.rbf_gamma = rbf_gamma
        self.degree = degree
        self.coef0 = coef0
        self.max_iter = max_iter
        self.verbose = verbose
        self.kernel = kernel
        self.multiclass_strategy = multiclass_strategy
        self.multiclass_ = None
        self.classes_ = None
        self._pairwise = self.kernel=='precomputed'

    def __kernel_definition__(self):
        """Select the kernel function
        
        Returns
        -------
        kernel : a callable relative to selected kernel
        """
        if hasattr(self.kernel, '__call__'):
            return self.kernel
        if self.kernel == 'rbf' or self.kernel == None:
            return lambda X,Y : rbf_kernel(X,Y,self.rbf_gamma)
        if self.kernel == 'poly':
            return lambda X,Y : polynomial_kernel(X, Y, degree=self.degree, gamma=self.rbf_gamma, coef0=self.coef0)
        if self.kernel == 'linear':
            return lambda X,Y : linear_kernel(X,Y)
        if self.kernel == 'precomputed':
            return lambda X,Y : X
    
    def fit(self, X, Y):
        
        """Fit the model according to the given training data
        
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Matrix of the examples, where
            n_samples is the number of samples and
            n_feature is the number of features
        
        Y : array-like, shape = [n_samples]
            array of the labels relative to X
        
        Returns
        -------
        self : object
            Returns self
        """
        X,Y = validation.check_X_y(X, Y, dtype=np.float64, order='C', accept_sparse='csr')
        #check_consistent_length(X,Y)
        check_classification_targets(Y)
        
        self.classes_ = np.unique(Y)
        if len(self.classes_) < 2:
            raise ValueError("The number of classes has to be almost 2; got ", len(self.classes_))
        
        if len(self.classes_) == 2:
            self.multiclass_ = False
            return self._fit(X,Y)
        else :
            self.multiclass_ = True
            if self.multiclass_strategy == 'ovo':
                return self._one_vs_one(X,Y)
            else :
                return self._one_vs_rest(X,Y)
        raise ValueError('This is a very bad exception...')
    
    def _one_vs_one(self,X,Y):
        self.cls = OneVsOneClassifier(KOMD(**self.get_params())).fit(X,Y)
        self.is_fitted = True
        return self
    
    def _one_vs_rest(self,X,Y):
        self.cls = OneVsRestClassifier(KOMD(**self.get_params())).fit(X,Y)
        self.is_fitted = True
        return self
        
    def _fit(self,X,Y):    
        self.X = X
        values = np.unique(Y)
        Y = [1 if l==values[1] else -1 for l in Y]
        self.Y = Y
        npos = len([1.0 for l in Y if l == 1])
        nneg = len([1.0 for l in Y if l == -1])
        gamma_unif = matrix([1.0/npos if l == 1 else 1.0/nneg for l in Y])
        YY = matrix(np.diag(list(matrix(Y))))

        Kf = self.__kernel_definition__()
        ker_matrix = matrix(Kf(X,X).astype(np.double))
        #KLL = (1.0 / (gamma_unif.T * YY * ker_matrix * YY * gamma_unif)[0])*(1.0-self.lam)*YY*ker_matrix*YY
        KLL = (1.0-self.lam)*YY*ker_matrix*YY
        LID = matrix(np.diag([self.lam * (npos * nneg / (npos+nneg))]*len(Y)))
        Q = 2*(KLL+LID)
        p = matrix([0.0]*len(Y))
        G = -matrix(np.diag([1.0]*len(Y)))
        h = matrix([0.0]*len(Y),(len(Y),1))
        A = matrix([[1.0 if lab==+1 else 0 for lab in Y],[1.0 if lab2==-1 else 0 for lab2 in Y]]).T
        b = matrix([[1.0],[1.0]],(2,1))
        
        solvers.options['show_progress'] = False#True
        solvers.options['maxiters'] = self.max_iter
        sol = solvers.qp(Q,p,G,h,A,b)
        self.gamma = sol['x']
        if self.verbose:
            print ('[KOMD]')
            print ('optimization finished, #iter = %d' % sol['iterations'])
            print ('status of the solution: %s' % sol['status'])
            print ('objval: %.5f' % sol['primal objective'])
            
        bias = 0.5 * self.gamma.T * ker_matrix * YY * self.gamma
        self.bias = bias
        self.is_fitted = True
        self.ker_matrix = ker_matrix
        return self
        
    def predict(self, X):
        """Perform classification on samples in X.
        
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Matrix containing new samples
        
        Returns
        -------
        y_pred : array, shape = [n_samples]
            The value of prediction for each sample
        """
        
        if self.is_fitted == False:
            raise NotFittedError("This KOMD instance is not fitted yet. Call 'fit' with appropriate arguments before using this method.")
        X = check_array(X, accept_sparse='csr', dtype=np.float64, order="C")
        if self.multiclass_ == True:
            return self.cls.predict(X)
        
        return np.array([self.classes_[1] if p >=0 else self.classes_[0] for p in self.decision_function(X)])

    def get_params(self, deep=True):
        # this estimator has parameters:
        return {"lam": self.lam, "kernel": self.kernel, "rbf_gamma":self.rbf_gamma,
                "degree":self.degree, "coef0":self.coef0, "max_iter":self.max_iter,
                "verbose":self.verbose, "multiclass_strategy":self.multiclass_strategy}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self,parameter,value)
        return self


    def decision_function(self, X):
        """Distance of the samples in X to the separating hyperplane.
        
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
        
        Returns
        -------
        Z : array-like, shape = [n_samples, 1]
            Returns the decision function of the samples.
        """
        
        if self.is_fitted == False:
            raise NotFittedError("This KOMD instance is not fitted yet. Call 'fit' with appropriate arguments before using this method.")
        X = check_array(X, accept_sparse='csr', dtype=np.float64, order="C")
        
        if self.multiclass_ == True:
            return self.cls.decision_function(X)
        
        Kf = self.__kernel_definition__()
        YY = matrix(np.diag(list(matrix(self.Y))))
        ker_matrix = matrix(Kf(X,self.X).astype(np.double))
        z = ker_matrix*YY*self.gamma
        z = z-self.bias
        return np.array(list(z))






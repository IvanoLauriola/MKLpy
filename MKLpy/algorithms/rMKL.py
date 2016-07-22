from sklearn.base import BaseEstimator, ClassifierMixin
from base import MKL
from sklearn.utils import check_array, check_consistent_length#, check_random_state
from sklearn.utils import column_or_1d, check_X_y
from sklearn.utils import validation
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import NotFittedError
from sklearn.utils.multiclass import check_classification_targets


from MKLpy.arrange import summation
from MKLpy.regularization import tracenorm
from MKLpy.lists import SFK_generator, HPK_generator
from MKLpy.utils.validation import check_kernel_list
import numpy as np

from cvxopt import matrix, solvers, mul, spdiag

class rMKL(BaseEstimator, ClassifierMixin, MKL):

    def __init__(self, C=1,D=1):
        self.C = C
        self.D = D
    

    def fit(self,K,Y):
        Ks = matrix(summation(K))
        YY = spdiag(matrix(Y))
        n = len(Y)
        K1 = (1.0-self.D)* YY*Ks*YY + spdiag([self.D] * n)
        #K2 = self.C * Ks * self.D
        K2 = (1.0-self.D)* self.C * Ks * self.D + spdiag([self.D] * n) * self.C
        P = 2*matrix([[K1[i,j] if i<n and j<n else K2[i-n,j-n] if i>=n and j>=n else 0.0   for i in range(2*n)] for j in range(2*n)])
        q = matrix([0.0 for i in range(2*n)])
        h = matrix([0.0]*(2*n),(2*n,1))
        G = -spdiag([1 for i in range(2*n)])
        A = matrix([[1.0 if i<n and Y[i]==+1 else 0.0 for i in range(2*n)],
                    [1.0 if i<n and Y[i]==-1 else 0.0 for i in range(2*n)],
                    [1.0 if i>=n else 0 for i in range(2*n)]]).T
        b = matrix([1.0,1.0,1.0])
        solvers.options['show_progress'] = False
        solvers.options['maxiters'] = 200
        sol = solvers.qp(P,q,G,h,A,b)
        x = sol['x']
        gamma = matrix(x[:n])
        alpha = matrix(x[n:2*n])
        #return
        weights = []
        for k in K:
            w = (gamma.T*YY*matrix(k)*YY*gamma)[0] + self.C * (alpha.T*matrix(k)*alpha)[0]
            weights.append(w)
        norm = sum([w for w in weights])
        self.weights = [w / norm for w in weights]
        
        #fine primo step
        
        Ks = matrix(summation(K,self.weights))
        K1 = (1.0-self.D)* YY*Ks*YY + spdiag([self.D] * n)
        #K2 = self.C * Ks * self.D
        K2 = (1.0-self.D)* self.C * Ks * self.D + spdiag([self.D] * n) * self.C
        P = 2*matrix([[K1[i,j] if i<n and j<n else K2[i-n,j-n] if i>=n and j>=n else 0.0   for i in range(2*n)] for j in range(2*n)])
        q = matrix([0.0 for i in range(2*n)])
        h = matrix([0.0]*(2*n),(2*n,1))
        G = -spdiag([1 for i in range(2*n)])
        A = matrix([[1.0 if i<n and Y[i]==+1 else 0.0 for i in range(2*n)],
                    [1.0 if i<n and Y[i]==-1 else 0.0 for i in range(2*n)],
                    [1.0 if i>=n else 0 for i in range(2*n)]]).T
        b = matrix([1.0,1.0,1.0])
        solvers.options['show_progress'] = False
        solvers.options['maxiters'] = 200
        sol = solvers.qp(P,q,G,h,A,b)
        x = sol['x']
        self.gamma = matrix(x[:n])
        self.alpha = matrix(x[n:2*n])
        self.Y=Y


        return self

    def decision_function(self, K):
         
        YY = spdiag(matrix(self.Y))
        ker_matrix = matrix(summation(K, self.weights))
        #z = ker_matrix*YY*self.gamma + self.C * ker_matrix*self.alpha
        #z = ker_matrix*YY*self.gamma + ker_matrix*self.alpha
        z = ker_matrix*YY*self.gamma
        return np.array(list(z))











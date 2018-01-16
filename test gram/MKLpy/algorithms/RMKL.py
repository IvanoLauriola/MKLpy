
'''
Margin and Radius Based Multiple Kernel Learning
'''
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
from base import MKL
from MKLpy.multiclass import OneVsOneMKLClassifier as ovoMKL, OneVsOneMKLClassifier as ovaMKL   #ATTENZIONE DUE OVO
from MKLpy.utils.exceptions import ArrangeMulticlassError
from MKLpy.lists import HPK_generator
from MKLpy.metrics import radius, margin
 
from cvxopt import matrix, spdiag, solvers
import numpy as np
 
from MKLpy.arrange import summation


class RMKL(BaseEstimator, ClassifierMixin, MKL):

    def __init__(self, estimator=SVC(), C=1, step=1, generator=HPK_generator(n=10), multiclass_strategy='ova', max_iter=500, tol=1e-7, verbose=False):
        super(self.__class__, self).__init__(estimator=estimator, generator=generator, multiclass_strategy=multiclass_strategy, how_to=summation, max_iter=max_iter, verbose=verbose)
        self.C = C
        self.step = step
        self.tol = tol


    def _arrange_kernel(self):
        Y = [1 if y==self.classes_[1] else -1 for y in self.Y]
        n = len(self.Y)
        R = np.array([radius(K) for K in self.KL])
        YY = matrix(np.diag(self.Y))

        actual_weights = np.ones(self.n_kernels) / (1.0 *self.n_kernels)    #current
        actual_ratio = None
        #weights = np.ones(self.n_kernels) / (1.0 *self.n_kernels)	#current

        self.ratios = []
        cstep = self.step
        for i in xrange(self.max_iter):
            ru2 = np.dot(actual_weights,R**2)
            C = self.C / ru2
            Kc = matrix(summation(self.KL,actual_weights))
            clf = SVC(C=C, kernel='precomputed').fit(Kc,Y)
            alpha = np.zeros(n)
            alpha[clf.support_] = clf.dual_coef_
            alpha = matrix(alpha)

            Q = Kc + spdiag( [ru2/self.C] * n )
            J = (-0.5 * alpha.T * YY * Q * YY * alpha)[0] + np.sum(alpha)
            grad = [(-0.5 * alpha.T * YY *(Kc+ spdiag([_r**2/self.C]*n)) * YY * alpha)[0] for _r in R]

            weights = actual_weights + cstep * np.array(grad)	#originalmente era un +
            weights = weights.clip(0.0)
            if weights.sum() == 0.0:
                #print i,'zero'
                cstep /= -2.0
                continue
            weights = weights/weights.sum()


            Kc = summation(self.KL,weights)
            new_ratio = radius(Kc)**2 / margin(Kc,Y)**2

            if actual_ratio and abs(new_ratio - actual_ratio)/n < self.tol:
                #completato
                #print i,'tol'
                self.ratios.append(new_ratio)
                actual_weights=weights
                #break;
            elif new_ratio <= actual_ratio or not actual_ratio:
                #tutto in regola
                #print i,'new step',new_ratio
                actual_ratio = new_ratio
                actual_weights = weights
                self.ratios.append(actual_ratio)
            else:
                #supero il minimo
                weights = actual_weights
                cstep /= -2.0
                #print i,'overflow',cstep
                continue
        self._steps = i+1
        
        self.weights = np.array(actual_weights)
        self.ker_matrix = summation(self.KL,self.weights)
        return self.ker_matrix
        return average(self.KL,weights)

    def get_params(self, deep=True):
        # this estimator has parameters:
        return {"C": self.C,
                "step":self.step,
                "tol":self.tol,
                "generator": self.generator, "max_iter":self.max_iter,
                "verbose":self.verbose, "multiclass_strategy":self.multiclass_strategy,
                'estimator':self.estimator}


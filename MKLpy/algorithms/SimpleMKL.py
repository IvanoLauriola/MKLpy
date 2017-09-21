
'''
Margin and Radius Based Multiple Kernel Learning
'''
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
from base import MKL
from MKLpy.multiclass import OneVsOneMKLClassifier as ovoMKL, OneVsOneMKLClassifier as ovaMKL   #ATTENZIONE DUE OVO
from MKLpy.utils.exceptions import ArrangeMulticlassError
from MKLpy.lists import HPK_generator
from MKLpy.metrics import margin
 
from cvxopt import matrix, spdiag, solvers
import numpy as np
 
from MKLpy.arrange import summation


class SimpleMKL(BaseEstimator, ClassifierMixin, MKL):

    def __init__(self, estimator=SVC(), C=1, step=1, generator=HPK_generator(n=10), multiclass_strategy='ova', max_iter=500, tol=1e-7, verbose=False):
        super(self.__class__, self).__init__(estimator=estimator, generator=generator, multiclass_strategy=multiclass_strategy, how_to=summation, max_iter=max_iter, verbose=verbose)
        self.C = C
        self.step = step
        self.tol = tol


    def _arrange_kernel(self):
        Y = [1 if y==self.classes_[1] else -1 for y in self.Y]
        n = len(self.Y)
        YY = spdiag(Y)
        actual_weights = np.ones(self.n_kernels) / (1.0 *self.n_kernels)    #current
        obj = None
        #weights = np.ones(self.n_kernels) / (1.0 *self.n_kernels)	#current
        print actual_weights
        self.objs = []
        cstep = self.step
        self.margin=[]
        for i in xrange(self.max_iter):
            Kc = summation(self.KL,actual_weights)
            #ottimizzo su alpha (posso prenderlo dallo step precedente....)
            clf = SVC(C=self.C, kernel='precomputed').fit(Kc,Y)
            alpha = np.zeros(n)
            alpha[clf.support_] = clf.dual_coef_
            alpha = matrix(alpha)

            #dati gli alpha ottimizzo sui pesi
            J = (-0.5 * alpha.T * YY * matrix(Kc) * YY * alpha)[0] + np.sum(alpha)
            grad = [(-0.5 * alpha.T * YY * matrix(_K) * YY * alpha)[0] for _K in self.KL]
            
            mu = np.argmax(actual_weights)	#all'inizio sono tutti uguali
            idx = np.where(actual_weights==max(actual_weights))
            mu = np.argmax(np.array(grad)[idx])
            D = [	0  if actual_weights[j]==0 and grad[j] - grad[mu] > 0	else \
            		-grad[j] + grad[mu] if actual_weights[j]>0 and j!=mu    else \
            		np.sum([grad[v]-grad[mu] for v in range(self.n_kernels) if grad[v]>0]) if j==mu 	else\
            		0
            	for j in range(self.n_kernels)]
            print 'd',D

            #aggiorno i pesi dato il gradiente
            weights = actual_weights + cstep * np.array(D)	#originalmente era un +
            weights = weights.clip(0.0)
            if weights.sum() == 0.0:
                print i,'zero',weights
                cstep /= 2.0
                continue
            weights = weights/weights.sum()

            #riottimizzo sugli alfa
            Kc = summation(self.KL,weights)
            clf = SVC(C=self.C, kernel='precomputed').fit(Kc,Y)
            alpha = np.zeros(n)
            alpha[clf.support_] = clf.dual_coef_
            alpha = matrix(alpha)
            new_obj = (-0.5 * alpha.T * YY * matrix(Kc) * YY * alpha)[0] + np.sum(alpha)
            if obj and abs(new_obj - obj)/n < self.tol:
                #completato
                #print i,'tol'
                self.objs.append(new_obj)
                actual_weights=weights
                print 'terminato',new_obj,obj
                break;
            elif new_obj <= obj or not obj:
                #tutto in regola
                #print i,'new step',new_ratio
                ma = margin(Kc,Y)
                if len(self.margin)>0 and  self.margin[-1] > ma:
                	continue
                self.margin.append(ma)
            
                obj = new_obj
                actual_weights = weights
                print actual_weights,obj
                self.objs.append(obj)
            else:
                #supero il minimo
                weights = actual_weights
                cstep /= 2.0
                print i,'overflow',cstep
                continue
        self._steps = i+1
        
        self.weights = np.array(actual_weights)
        self.ker_matrix = summation(self.KL,self.weights)

        return self.ker_matrix
        #return average(self.KL,weights)

    def get_params(self, deep=True):
        # this estimator has parameters:
        return {"C": self.C,
                "step":self.step,
                "tol":self.tol,
                "generator": self.generator, "n_kernels": self.n_kernels, "max_iter":self.max_iter,
                "verbose":self.verbose, "multiclass_strategy":self.multiclass_strategy,
                'estimator':self.estimator}


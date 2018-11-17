# -*- coding: latin-1 -*-
"""
"""
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
from MKLpy.algorithms.base import MKL
from MKLpy.multiclass import OneVsOneMKLClassifier as ovoMKL, \
    OneVsOneMKLClassifier as ovaMKL  # ATTENZIONE DUE OVO
from MKLpy.utils.exceptions import ArrangeMulticlassError
from MKLpy.lists import HPK_generator

from cvxopt import matrix, spdiag, solvers
import numpy as np
import time, sys
from MKLpy.arrange import summation


def radius(K, lam=0, init_sol=None):
    n = K.shape[0]
    K = matrix(K)
    P = 2 * ((1 - lam) * K + spdiag([lam] * n))
    p = -matrix([K[i, i] for i in range(n)])
    G = -spdiag([1.0] * n)
    h = matrix([0.0] * n)
    A = matrix([1.0] * n).T
    b = matrix([1.0])
    solvers.options['show_progress'] = False
    sol = solvers.qp(P, p, G, h, A, b, initvals=init_sol)
    radius2 = (-p.T * sol['x'])[0] - (sol['x'].T * K * sol['x'])[0]
    return sol, radius2


def margin(K, Y, lam=0, init_sol=None):
    n = len(Y)
    YY = spdiag(list(Y))
    K = matrix(K)
    lambdaDiag = spdiag([lam] * n)
    P = 2 * ((1 - lam) * (YY * K * YY) + lambdaDiag)
    p = matrix([0.0] * n)
    G = -spdiag([1.0] * n)
    h = matrix([0.0] * n)
    A = matrix([[1.0 if Y[i] == +1 else 0 for i in range(n)],
                [1.0 if Y[j] == -1 else 0 for j in range(n)]]).T
    b = matrix([[1.0], [1.0]], (2, 1))
    solvers.options['show_progress'] = False
    sol = solvers.qp(P, p, G, h, A, b, initvals=init_sol)
    margin2 = sol['dual objective'] - (sol['x'].T * lambdaDiag * sol['x'])[0]
    return sol, margin2


class GRAM(BaseEstimator, ClassifierMixin, MKL):

    def __init__(self, estimator=SVC(), step=1.0, generator=HPK_generator(n=10),
                 multiclass_strategy='ova', max_iter=10000, tol=1e-9, verbose=False, n_folds=1,
                 random_state=42, lam=0):
        super(self.__class__, self).__init__(estimator=estimator, generator=generator,
                                             multiclass_strategy=multiclass_strategy,
                                             how_to=summation, max_iter=max_iter, verbose=verbose)
        self.step = step
        self.lam = lam
        self.tol = tol
        self.n_folds = n_folds
        self.random_state = random_state

    def _arrange_kernel(self):

        Y = [1 if y == self.classes_[1] else -1 for y in self.Y]
        YY = spdiag(Y)
        Y = np.array(Y)
        nn = len(Y)
        nk = self.n_kernels

        idx_e = range(nn)
        np.random.seed(self.random_state)
        np.random.shuffle(idx_e)

        contexts = [{'idx'  : idx_e[i::self.n_folds],
                     'alpha': None,
                     'gamma': None
                     } for i in range(self.n_folds)]

        beta = [0.0] * nk
        mu = np.exp(np.array(beta) - max(beta))
        mu /= mu.sum()

        Kc = summation(self.KL, mu)
        initial_ratio = []
        for context in contexts:
            idx = context['idx']
            context['alpha'], r2 = radius(Kc[idx][:, idx], lam=self.lam)
            context['gamma'], m2 = margin(Kc[idx][:, idx], Y[idx], lam=self.lam)
            initial_ratio.append((r2 / m2) / len(context['idx']))
        initial_ratio = np.mean(initial_ratio)

        self._ratios = [initial_ratio]

        cstep = self.step
        self._converg = False
        self._steps = 0

        while (not self._converg and (self._steps < self.max_iter)):
            self._steps += 1

            new_beta = beta[:]
            new_ratio = []
            for context in contexts:
                idx = context['idx']
                new_beta = self.update_grad(Kc, YY, new_beta, context, cstep)
                new_mu = np.exp(new_beta - max(new_beta))
                new_mu /= new_mu.sum()
                new_Kc = summation(self.KL, new_mu)
                try:
                    new_alpha, r2 = radius(new_Kc[idx][:, idx], lam=self.lam,
                                           init_sol=context['alpha'].copy())
                    new_gamma, m2 = margin(new_Kc[idx][:, idx], Y[idx], lam=self.lam,
                                           init_sol=context['gamma'].copy())
                except:
                    new_alpha, r2 = radius(Kc[idx][:, idx], lam=self.lam)
                    new_gamma, m2 = margin(Kc[idx][:, idx], Y[idx], lam=self.lam)
                new_ratio.append((r2 / m2) / len(idx))
            new_ratio = np.mean(new_ratio)

            if not self._ratios or new_ratio < self._ratios[-1]:  # or self._steps < 2:
                beta = new_beta[:]
                mu = np.exp(beta - max(beta))
                mu /= mu.sum()
                Kc = new_Kc
                self._ratios.append(new_ratio)
                print
                self._steps, new_ratio, 'ok', cstep
            else:
                cstep /= 10.0
                print
                self._steps, new_ratio, 'peggiorativo', self._ratios[-1], cstep
                if cstep < 1e-10:
                    self._converg = True
                continue

        self.weights = np.array(mu)
        self.ker_matrix = summation(self.KL, self.weights)
        return self.ker_matrix

    def update_grad(self, Kc, YY, _beta, context, cstep):
        idx = context['idx']
        alpha = context['alpha']
        gamma = context['gamma']
        Ks = Kc[idx][:, idx]
        YYs = YY[idx, idx]
        eb = np.exp(np.array(_beta))

        a = np.array(
            [1.0 - (alpha['x'].T * matrix(K[idx][:, idx]) * alpha['x'])[0] for K in self.KL])
        b = np.array(
            [(gamma['x'].T * YYs * matrix(K[idx][:, idx]) * YYs * gamma['x'])[0] for K in self.KL])
        den = [np.dot(eb, b) ** 2] * self.n_kernels
        num = [eb[r] * (a[r] * np.dot(eb, b) - b[r] * np.dot(eb, a)) for r in range(self.n_kernels)]

        new_beta = np.array([_beta[k] - cstep * (num[k] / den[k]) for k in range(self.n_kernels)])
        new_beta = new_beta - new_beta.max()
        return new_beta

    def get_params(self, deep=True):
        # this estimator has parameters:
        return {"step"     : self.step,
                "tol"      : self.tol,
                "generator": self.generator, "max_iter": self.max_iter,
                "verbose"  : self.verbose, "multiclass_strategy": self.multiclass_strategy,
                'estimator': self.estimator}

from itertools import product
from sklearn.model_selection._search import BaseSearchCV as skbs
from MKLpy.model_selection import cross_val_score
import sys
import warnings
import numpy as np

class BaseSearchCV(skbs):

    def __init__(self, estimator, base=None, scoring=None,
                 fit_params=None, n_jobs=1, iid=True,
                 refit=True, cv=None, verbose=0, pre_dispatch='2*n_jobs',
                 error_score='raise', return_train_score=True):
        raise NotImplementedError('Not implemented yet')
        self.scoring = scoring
        self.estimator = estimator
        self.n_jobs = n_jobs
        self.fit_params = fit_params if fit_params is not None else {}
        self.iid = iid
        self.refit = refit
        self.cv = cv
        self.verbose = verbose
        self.pre_dispatch = pre_dispatch
        self.error_score = error_score
        self.return_train_score = return_train_score
        self.base = base if base else estimator.base


    def fit(Klist, Y):
        warnings.warn("it works only with accuracy score")
        raise NotImplementedError('Not implemented yet')
        
        estimator = self.estimator
        base = self.base
        cv = check_cv(self.cv,Y)

        params_mkl  = list(_get_param_iterator())
        params_base = list(_get_param_iterator_base())

        self.scores = np.zeros(len(params_mkl)) if self.auto_classifier else np.zeros( (len(params_mkl),len(params_base)))
        self.best_score_     = None
        self.best_estimator_ = None #tupla se ho un base
        self.best_params_    = None #tupla se ho un base
        self.best_index_     = None #tupla se ho un base
        self.scorer_         = check_scoring(self.estimator, scoring=self.scoring)
        self.n_splits_       = self.cv.n_splits_
        
        if self.verbose:
            print "tuning params, %d combinations" % (len(params_mkl)) if self.autoclassifier else \
                  "tuning params, %d x %d combinations considering MKL and base learner" % (len(params_mkl), len(params_base))
            print "# folds: %d" % (self.n_splits_)
            
            
        if self.auto_classifier:    #uso solo l'algoritmo di MKL
            for i,p in enumerate(params_mkl):
                clone = self.estimator.__class__()
                clone.set_params(p)
                res = cross_validation(Klist, Y, clone, cv=self.cv,score='accuracy')     #attenzione, solo con accuracy
                _set_best(res,i,clone,p)
        else:   #MKL + base learner diverso
            for i,p_mkl in enumerate(params_mkl):
                clone_mkl = self.estimator.__class__()
                clone_MKL.set_params(p_mkl)
                for j,p_base in enumerate(params_base):
                    clone_base = self.base.__class__()
                    clone_base.set_params(p_base)
                    res = cross_validation(Klist, Y, clone_mkl, base=clone_base, cv=self.cv,score='accuracy')
                    _set_best(res,(i,j),(clone_mkl,clone_base),(p_mkl,p_base))
        
        def _set_best(res,idx,estimator,params):   #risultato, indice
            v,s = np.mean(res),np.std(res)
            scores[idx] = v
            if not self.best_score or v > self.best_score_:
                self.best_score_     = v
                self.best_estimator_ = estimator
                self.best_params_    = params
                self.best_index_     = idx
        
        if self.verbose:
            print "best model:"
            print "   params:",self.best_params if self.auto_classifier else \
                  "   params MKL :", self.best_params[0],"\n   params base:", self.best_params[1]
        
        self.cv_results_ = None
        if self.refit:
            if self.auto_classifier:
                self.best_estimator_ = self.best_estimator.fit(Klist,Y)
            else:
                Kc = self.best_estimator[0].arrange_kernels(Klist,Y)
                self.best_estimator[1].fit(Kc,Y)
            print "refit: done"
                
    
    def predict(Kte):
        if self.auto_classifier:
            return self.best_estimator.predict(Kte)
        else:
            w = self.best_estimator_[0].weights
            f = self.best_estimator_[0].how_to
            return self.best_estimator_[1].predict(f(Kte,w))
        
    
    def decision_function(Kte):
        if self.auto_classifier:
            return self.best_estimator.decision_function(Kte)
        else:
            w = self.best_estimator_[0].weights
            f = self.best_estimator_[0].how_to
            return self.best_estimator_[1].decision_function(f(Kte,w))
        
    



class GridSearchCV(BaseSearchCV):
    def __init__(self, estimator, param_grid_MKL, base=None, param_grid_base=None, scoring=None, fit_params=None,
                 n_jobs=1, iid=True, refit=True, cv=None, verbose=0,
                 pre_dispatch='2*n_jobs', error_score='raise',
                 return_train_score=True):
        super(GridSearchCV, self).__init__(
            estimator=estimator, base=base, scoring=scoring, fit_params=fit_params,
            n_jobs=n_jobs, iid=iid, refit=refit, cv=cv, verbose=verbose,
            pre_dispatch=pre_dispatch, error_score=error_score,
            return_train_score=return_train_score)
        self.param_grid = param_grid_MKL
        self.param_grid_base = param_grid_base
        _check_param_grid(param_grid)
        _check_param_grid(param_grid_base)

        # True -> uso solo un algoritmo
        self.auto_classifier = not (base and param_grid_base) #se manca qualcosa uso solo MKL
    
    def _get_param_iterator(self):
        """Return ParameterGrid instance for the given param_grid"""
        return ParameterGrid(self.param_grid)

    def _get_param_iterator_base(self):
        '''i parametri del base learner'''
        return ParameterGrid(self.param_grid_base)

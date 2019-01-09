from sklearn.metrics import accuracy_score,roc_auc_score
from sklearn.model_selection import StratifiedKFold as KFold
import numpy as np


def __def_score__(score):
    #return roc_auc_score
    if score in ['auc','AUC','roc_auc']:
        return roc_auc_score
    elif score=='accuracy' or True:
        return accuracy_score



#data una lista di kernel, un classificatore ed i parametri faccio cv
def cross_val_score(KL, Y, estimator, cv=None, n_folds=3, score='roc_auc', random_state=None):
    return _cross_val (KL, Y, estimator, 'decision_function', cv, n_folds, score, random_state)

def cross_val_predict(KL, Y, estimator, cv=None, n_folds=3, score='accuracy', random_state=None):
    return _cross_val (KL, Y, estimator, 'predict', cv, n_folds, score, random_state)


def _cross_val(KL, Y, estimator, f, cv=None, n_folds=3, score='roc_auc', random_state=None):
    f = getattr(estimator,f)
    n = len(Y)
    score = __def_score__(score)
    cv   = cv or KFold(n_folds,random_state = random_state)
    scores = []
    for train,test in cv.split(Y,Y):
        KLtr = [K[train][:,train]for K in KL]
        KLte = [K[test][:,train]for K in KL]
        clf = estimator.fit(KLtr,Y[train])
        y = f(KLte)
        scores.append(score(Y[test],y))
    return scores

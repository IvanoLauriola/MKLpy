from sklearn.metrics import accuracy_score,roc_auc_score
from sklearn.model_selection import StratifiedKFold as KFold
import numpy as np
from splits import cv3


def __def_score__(score):
    return roc_auc_score
    if score in ['auc','AUC']:
        return roc_auc_score
    elif score=='accuracy' or True:
        return accuracy_score



#data una lista di kernel, un classificatore ed i parametri faccio cv
def cross_val_score(KL, Y, estimator, cv=None, n_folds=3, score='accuracy'):
    n = Y.shape[0]
    score = __def_score__(score)
    cv   = cv   if cv   else KFold(n_folds).split(Y,Y)
    scores = []
    for train,test in cv:
        tr,te = cv3(train,test,KL.shape[0])
        clf = estimator.fit(KL[tr],Y[train])
        y = clf.predict(KL[te])
        scores.append(score(Y[test],y))
    return scores

    

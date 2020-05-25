from sklearn.metrics import accuracy_score,roc_auc_score
from sklearn.model_selection import StratifiedKFold as KFold
from sklearn import model_selection as skms
import numpy as np
import torch


def __def_score__(score):
    '''internal check'''
    score = score.lower()
    if score == 'roc_auc':
    	return roc_auc_score, 'decision_function'
    elif score == 'accuracy':
    	return accuracy_score, 'predict'
    elif score == 'f_score':
    	return f1_score, 'predict'
    else:
    	raise ValueError('%s is not a valid metric. Valid metrics are \'roc_auc\', \'accuracy\', or \'f_score\'.' % score)



def cross_val_score(KL, Y, estimator, cv=None, n_folds=3, scoring='accuracy', random_state=None, shuffle=True):
    '''performs the cross validation'''

    scorer, f = __def_score__(scoring)
    f = getattr(estimator,f)
    n = len(Y)
    cv   = cv or KFold(n_folds, random_state=random_state, shuffle=shuffle)
    results = []
    for train,test in cv.split(Y,Y):
        KLtr = [K[train][:,train] for K in KL]
        KLte = [K[test ][:,train] for K in KL]
        clf = estimator.fit(KLtr,Y[train])
        y = f(KLte)
        results.append(scorer(Y[test],y))
    return results





def train_test_split(KL, Y, train_size=None, test_size=None, random_state=None, shuffle=True):
    '''returns two kernel lists, for train and test'''

    idx = range(len(Y))
    train,test = skms.train_test_split(idx, 
        train_size=train_size, 
        test_size=test_size, 
        random_state=random_state, 
        shuffle=shuffle)
    KL_tr = [K[train][:,train] for K in KL]
    KL_te = [K[test ][:,train] for K in KL]
    return KL_tr, KL_te, Y[train], Y[test]


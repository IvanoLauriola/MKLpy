import numpy as np
from sklearn import model_selection as skms



def train_test_split(KL, Y, train_size=None, test_size=None, random_state=None):
    '''returns two kernel lists, for train and test'''
    train,test = skms.train_test_split(range(len(Y)),train_size=train_size,test_size=test_size,random_state=random_state)
    KL_tr = [K[train][:,train] for K in KL]
    KL_te = [K[test ][:,train] for K in KL]
    return KL_tr, KL_te, np.array(Y)[train], np.array(Y)[test]


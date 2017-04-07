import numpy as np
from sklearn import model_selection as skms



def train_test_split(KL, Y, train_size=None, test_size=None, random_state=None):
    '''returns two kernel lists, for train and test'''
    train,test = skms.train_test_split(range(Y.shape[0]),train_size=train_size,test_size=test_size)
    tr,te = cv3(train,test,KL.shape[0])
    return Klist[tr],Klist[te],Y[train],Y[test]


def cv3(train,test,n):
    n = np.arange(n)
    return np.ix_(n,train,train),np.ix_(n,test,train)
    #provare con np.ix_(:,train,train),np.ix_(:,test,train)


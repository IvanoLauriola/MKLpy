import numpy as np
from MKLpy.metrics import trace

'''SOLO OPERAZIONI CHE ALTERANO LA RAPPRESENTAZIONE DEL KERNEL (o dei dati)'''

def normalization(X, axis=0):
    #0 = rows, 1 = cols (TODO)
    for idrow in xrange(X.shape[0]):
        X[idrow,:] = X[idrow,:] / np.linalg.norm(X[idrow,:])
    return X


def kernel_normalization(K):
    n = K.shape[0]
    d = np.array([[K[i,i] for i in range(n)]])
    Kn = K / np.sqrt(np.dot(d.T,d))
    return Kn
    

def tracenorm(K):
    #return sum([K[i,i] for i in range(k.shape[0])]) / K.shape[0]
    trn = trace(K) / K.shape[0]
    return K / trn



def rescale(X):
    X = rescale_01(X)
    return (X * 2) - 1


def rescale_01(X):
    d = X.shape[1]
    for i in range(d):
        mi_v = min(X[:,i])
        ma_v = max(X[:,i])
        if mi_v!=ma_v:
            X[:,i] = (X[:,i] - mi_v)/(ma_v-mi_v)
    return X


def centering(X):
    n = X.shape[0]
    uno = np.ones((n,1))
    Xm = 1.0/n * np.dot(uno.T,X)
    return X - np.dot(uno,Xm)


def kernel_centering(K):
    N = K.shape[0] * 1.0
    I = np.ones(K.shape)
    C = np.diag(np.ones(N)) - (1/N * I)
    Kc = np.dot(np.dot(C , K) , C)
    return Kc

def edit(Klist,f):
    return np.array([f(K) for K in Klist])







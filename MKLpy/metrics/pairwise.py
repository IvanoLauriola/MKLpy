import numpy as np
from math import floor
from sklearn.metrics.pairwise import cosine_similarity

def HPK_kernel(X, T=None, degree=2):
    if degree < 1:
        raise ValueError('degree must be greather than 0')
    if degree != floor(degree):
        raise ValueError('degree must be int')
    if T==None:
        T = X
    return np.dot(X,T)**degree


def SSK_kernel(X, T=None, k=2):
    return


def identity_kernel(X, T=None, s=1):
    if T is None:
        T = X
    K = (cosine_similarity(T,X) >= s*0.999999) * 1.0
    return K


def ideal_kernel(y, w=None):
    if w==None:
        w = y
    return np.dot(np.array([y]).T,np.array([w]))


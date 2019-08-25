'''

This is a snippet of code showing how to train a MKL algorithm

Author: Ivano Lauriola, ivano.lauriola@phd.unipd.it

'''





#load data
print ('loading \'iris\' dataset...', end='')
from sklearn.datasets import load_iris
import numpy as np
ds = load_iris()
X,Y = ds.data, ds.target
classes = np.unique(Y)
print ('done [%d classes]' % len(classes))

'''
WARNING: be sure that your matrix is not sparse! EXAMPLE:
from sklearn.datasets import load_svmlight_file
X,Y = load_svmlight_file(...)
X = X.toarray()
'''

#compute homogeneous polynomial kernels with degrees 0,1,2,...,10.
print ('computing Homogeneous Polynomial Kernels...', end='')
from MKLpy.metrics import pairwise
KL = [pairwise.homogeneous_polynomial_kernel(X, degree=d) for d in range(1,4)]
print ('done')


#MKL algorithms
from MKLpy.algorithms import EasyMKL
print ('training EasyMKL...', end='')
clf = EasyMKL(lam=0.1, multiclass_strategy='ovo').fit(KL,Y)		#combining kernels with the EasyMKL algorithm
#multiclass_strategy should be 'ovo' for one-vs-one decomposition strategy, and 'ova' for one-vs-all/rest strategy
print ('done')

print (clf.weights)

'''

This is a snippet of code showing how to train a MKL algorithm

Author: Ivano Lauriola, ivano.lauriola@phd.unipd.it

'''

from sklearn.datasets import load_iris
ds = load_iris()
X,Y = ds.data, ds.target


from MKLpy.preprocessing import normalization
X = normalization(X)


from sklearn.model_selection import train_test_split
Xtr,Xte,Ytr,Yte = train_test_split(X,Y, test_size=.5, random_state=42)


from MKLpy.metrics import pairwise
from MKLpy.utils.matrices import identity_kernel
import numpy as np

#making 20 homogeneous polynomial kernels.
#I suggest to add the identity kernel in order to make the GRAM initial solution easily separable
#if the initial sol is not separable, GRAM may not work well
KLtr = [pairwise.homogeneous_polynomial_kernel(Xtr, degree=d) for d in range(1,21)] + [identity_kernel(len(Ytr))]
KLte = [pairwise.homogeneous_polynomial_kernel(Xte,Xtr, degree=d) for d in range(1,21)]
KLte.append(np.zeros(KLte[0].shape))


from MKLpy.algorithms import GRAM
from sklearn.svm import SVC

#play with max iter (reduce the number if the problem is big) and learning rate!
clf = GRAM(max_iter=1000, learner=SVC(C=1000), learning_rate=1).fit(KLtr,Ytr)
print (clf.weights)

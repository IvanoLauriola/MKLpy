'''

This is a snippet of code showing how to train a multiclass MKL algorithm

Author: Ivano Lauriola, ivano.lauriola@phd.unipd.it

'''





#load data
print ('loading \'iris\' dataset...', end='')
from sklearn.datasets import load_iris as load
ds = load()
X,Y = ds.data, ds.target
print ('done')

'''
WARNING: be sure that your matrix is not sparse! EXAMPLE:
from sklearn.datasets import load_svmlight_file
X,Y = load_svmlight_file(...)
X = X.toarray()
'''

#preprocess data
print ('preprocessing data...', end='')
from MKLpy.preprocessing import normalization, rescale_01
X = rescale_01(X)	#feature scaling in [0,1]
X = normalization(X) #||X_i||_2^2 = 1

#train/test split
from sklearn.model_selection import train_test_split
Xtr,Xte,Ytr,Yte = train_test_split(X,Y, test_size=.3, random_state=42, shuffle=True)
print ('done')


#compute homogeneous polynomial kernels with degrees 0,1,2,...,10.
print ('computing Homogeneous Polynomial Kernels...', end='')
from MKLpy.metrics import pairwise
KLtr = [pairwise.homogeneous_polynomial_kernel(Xtr, degree=d) for d in range(11)]
KLte = [pairwise.homogeneous_polynomial_kernel(Xte,Xtr, degree=d) for d in range(11)]
print ('done')


#MKL algorithms
from MKLpy.algorithms import AverageMKL, EasyMKL


print ('training EasyMKL with one-vs-all multiclass strategy...', end='')
from sklearn.svm import SVC
base_learner = SVC(C=0.1)
clf = EasyMKL(lam=0.1, multiclass_strategy='ova', learner=base_learner).fit(KLtr,Ytr)
from MKLpy.multiclass import OneVsRestMKLClassifier, OneVsOneMKLClassifier
print ('done')
print ('the combination weights are:')
for sol in clf.solution:
	print ('(%d vs all): ' % sol, clf.solution[sol].weights)
	
#evaluate the solution
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np
y_pred = clf.predict(KLte)					#predictions
y_score = clf.decision_function(KLte)		#rank
accuracy = accuracy_score(Yte, y_pred)
print ('Accuracy score: %.3f' % (accuracy))



print ('training EasyMKL with one-vs-one multiclass strategy...', end='')
clf = EasyMKL(lam=0.1, multiclass_strategy='ovo', learner=base_learner).fit(KLtr,Ytr)
print ('done')
print ('the combination weights are:')
for sol in clf.solution:
	print ('(%d vs %d): ' % (sol[0], sol[1]), clf.solution[sol].weights)







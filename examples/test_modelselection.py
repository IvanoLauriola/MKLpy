'''

This is a snippet of code showing ho to select the hyper-parameters
of a MKL method using boolean kernels

Author: Ivano Lauriola, ivano.lauriola@phd.unipd.it

'''
import MKLpy




#load data
print ('loading \'iris\' (multiclass) dataset...', end='')
from sklearn.datasets import load_iris
ds = load_iris()
X,Y = ds.data, ds.target

'''
WARNING: be sure that your matrix is not sparse! EXAMPLE:
from sklearn.datasets import load_svmlight_file
X,Y = load_svmlight_file(...)
X = X.toarray()
'''

print ('done')

#preprocess data
print ('preprocessing data...', end='')
#boolean kernels can be applied on binary-valued data, i.e. {0,1}.
#in this example, we binarize a real-valued dataset
from MKLpy.preprocessing import binarization
binarizer = binarization.AverageBinarizer()
Xbin = binarizer.fit_transform(X,Y)
print ('done')

#compute normalized homogeneous polynomial kernels with degrees 0,1,2,...,10.
print ('computing monotone Conjunctive Kernels...', end='')
from MKLpy.metrics import pairwise
from MKLpy.preprocessing import kernel_normalization
#WARNING: the maximum arity of the conjunctive kernel depends on the number of active variables for each example,
# that is 4 in the case of iris dataset binarized
KL = [kernel_normalization(pairwise.monotone_conjunctive_kernel(Xbin, c=c)) for c in range(5)]
print ('done')

#train/test KL split (N.B. here we split a kernel list directly)
from MKLpy.model_selection import train_test_split
KLtr,KLte,Ytr,Yte = train_test_split(KL, Y, test_size=.3, random_state=42)

#MKL algorithms
from MKLpy.algorithms import EasyMKL, KOMD	#KOMD is not a MKL algorithm but a simple kernel machine like the SVM
from MKLpy.model_selection import cross_val_score, cross_val_predict
from sklearn.svm import SVC
import numpy as np
print ('tuning lambda for EasyMKL...', end='')
base_learner = SVC(C=10000)	#simil hard-margin svm
best_results = {}
for lam in [0, 0.01, 0.1, 0.2, 0.9, 1]:	#possible lambda values for the EasyMKL algorithm
	#MKLpy.model_selection.cross_val_predict performs the cross validation automatically, it optimizes the accuracy
	#the counterpart cross_val_score optimized the roc_auc_score (use score='roc_auc')
	#WARNING: these functions will change in the next version
	scores = cross_val_predict(KLtr, Ytr, EasyMKL(estimator=base_learner, lam=lam), n_folds=5, score='accuracy')
	acc = np.mean(scores)
	if not best_results or best_results['score'] < acc:
		best_results = {'lam' : lam, 'score' : acc}
#evaluation on the test set
from sklearn.metrics import accuracy_score
print ('done')
clf = EasyMKL(estimator=base_learner, lam=best_results['lam']).fit(KLtr,Ytr)
y_pred = clf.predict(KLte)
accuracy = accuracy_score(Yte, y_pred)
print ('accuracy on the test set: %.3f, with lambda=%.2f' % (accuracy, best_results['lam']))

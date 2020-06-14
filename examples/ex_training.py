'''

This is a snippet of code showing how to train a MKL algorithm

Author: Ivano Lauriola, ivano.lauriola@phd.unipd.it

'''





#load data
print ('loading \'breast cancer\' dataset...', end='')
from sklearn.datasets import load_breast_cancer
ds = load_breast_cancer()
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
Xtr,Xte,Ytr,Yte = train_test_split(X,Y, test_size=.25, random_state=42)
print ('done')


#compute homogeneous polynomial kernels with degrees 0,1,2,...,10.
print ('computing Homogeneous Polynomial Kernels...', end='')
from MKLpy.metrics import pairwise
KLtr = [pairwise.homogeneous_polynomial_kernel(Xtr, degree=d) for d in range(11)]
KLte = [pairwise.homogeneous_polynomial_kernel(Xte,Xtr, degree=d) for d in range(11)]
print ('done')


#MKL algorithms
from MKLpy.algorithms import AverageMKL, EasyMKL, KOMD	#KOMD is not a MKL algorithm but a simple kernel machine like the SVM
print ('training AverageMKL...', end='')
clf = AverageMKL().fit(KLtr,Ytr)	#a wrapper for averaging kernels
print ('done')
K_average = clf.solution.ker_matrix	#the combined kernel matrix


print ('training EasyMKL...', end='')
clf = EasyMKL(lam=0.1).fit(KLtr,Ytr)		#combining kernels with the EasyMKL algorithm
#lam is a hyper-parameter in [0,1]
print ('done')
print ('the combination weights are:')
print (clf.solution.weights)

#evaluate the solution
from sklearn.metrics import accuracy_score, roc_auc_score
y_pred = clf.predict(KLte)					#predictions
y_score = clf.decision_function(KLte)		#rank
accuracy = accuracy_score(Yte, y_pred)
roc_auc = roc_auc_score(Yte, y_score)
print ('Accuracy score: %.3f, roc AUC score: %.3f' % (accuracy, roc_auc))


#select the base-learner
#MKL algorithms use a hard-margin SVM as base learned (or KOMD in the case of EasyMKL).
#It is possible to define a different base learner
from sklearn.svm import SVC
base_learner = SVC(C=0.1)
print ('training EasyMKL with a soft-SVM...', end='')
clf = EasyMKL(learner=base_learner)
clf = clf.fit(KLtr,Ytr)
print ('done')
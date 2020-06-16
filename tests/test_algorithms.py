


import sys
sys.path.append("../MKLpy")

import unittest
import torch
from sklearn.datasets import load_breast_cancer, load_iris, load_digits
from sklearn.metrics import pairwise as pairwise_sk
from sklearn.model_selection import train_test_split, LeaveOneOut
from sklearn.exceptions import NotFittedError
from sklearn.svm import SVC
from MKLpy import preprocessing
from MKLpy import callbacks
from MKLpy import metrics
from MKLpy import multiclass
from MKLpy.scheduler import ReduceOnWorsening
from MKLpy.metrics import pairwise as pairwise_mk
from MKLpy import algorithms
from MKLpy.utils.exceptions import SquaredKernelError, InvalidKernelsListError, BinaryProblemError
from MKLpy.utils import misc
from MKLpy.generators import HPK_generator, RBF_generator, Lambda_generator, Multiview_generator
from MKLpy.arrange import average, multiplication, summation
from MKLpy.model_selection import train_test_split as tts, cross_val_score
from MKLpy.preprocessing.binarization import AverageBinarizer





def matNear(a,b, eps=1e-7):
	#return np.isclose(a,b, atol=eps).min()
	a = torch.tensor(a) if type(a) != torch.Tensor else a
	b = torch.tensor(b) if type(b) != torch.Tensor else b
	a = a.type(torch.float64)
	b = b.type(torch.float64)
	return torch.allclose(a, b, atol=eps)



class TestKOMD(unittest.TestCase):

	def setUp(self):
		data = load_breast_cancer()
		self.Xtr, self.Xte, self.Ytr, self.Yte = train_test_split(data.data, data.target, shuffle=True, train_size=50)

	def test_fit(self):
		K = self.Xtr @ self.Xtr.T
		clf = algorithms.KOMD().fit(self.Xtr, self.Ytr+1)
		clf1 = algorithms.KOMD(kernel='precomputed').fit(K, self.Ytr+1)
		params = clf1.get_params()
		self.assertTrue(params['lam'] == 0.1)
		self.assertTrue(params['kernel'] == 'precomputed')
		clf2 = algorithms.KOMD(kernel='rbf', rbf_gamma=.01).fit(self.Xtr, self.Ytr+1)
		clf3 = algorithms.KOMD(kernel='poly', degree=2).fit(self.Xtr, self.Ytr+1)
		clf4 = algorithms.KOMD(kernel='linear').fit(self.Xtr, self.Ytr+1)
		y1 = clf1.decision_function(self.Xte @ self.Xtr.T)
		y2 = clf4.decision_function(self.Xte)
		self.assertListEqual(y1.tolist(), y2.tolist())
		y1 = clf1.predict(self.Xte @ self.Xtr.T)
		y2 = clf4.predict(self.Xte)
		self.assertListEqual(y1.tolist(), y2.tolist())

	def test_multiclass(self):
		data = load_iris()
		Xtr, Xte, Ytr, Yte = train_test_split(data.data, data.target, shuffle=True, train_size=100)
		Ktr = (Xtr @ Xtr.T) **2
		Kte = (Xte @ Xtr.T) **2
		y1 = algorithms.KOMD(kernel='poly', degree=2, coef0=0, rbf_gamma=1).fit(Xtr, Ytr).predict(Xte)
		y2 = algorithms.KOMD(kernel='precomputed').fit(Ktr, Ytr).predict(Kte)
		self.assertListEqual(y1.tolist(), y2.tolist())





class TestMKL(unittest.TestCase):

	def setUp(self):
		data = load_breast_cancer()
		self.Xtr, self.Xte, self.Ytr, self.Yte = train_test_split(data.data, data.target, shuffle=True, train_size=50)
		self.Xtr = preprocessing.normalization(self.Xtr)
		self.Xte = preprocessing.normalization(self.Xte)
		self.KLtr = [pairwise_mk.homogeneous_polynomial_kernel(self.Xtr, degree=d) for d in range(5,11)] + [misc.identity_kernel(len(self.Xtr))]#.Double()]
		self.KLte = [pairwise_mk.homogeneous_polynomial_kernel(self.Xte, self.Xtr, degree=d) for d in range(5,11)] + [torch.zeros(len(self.Xte), len(self.Xtr))]#, dtype=torch.double)]
		self.KLtr_g = HPK_generator(self.Xtr, degrees=range(5,11), include_identity=True)
		self.KLte_g = HPK_generator(self.Xte, self.Xtr, degrees=range(5,11), include_identity=True)

	def base_evaluation(self, clf):
		clf = clf.fit(self.KLtr, self.Ytr)
		w = clf.solution.weights
		ker = clf.solution.ker_matrix
		self.assertRaises(ValueError, clf.fit, self.Xtr, self.Ytr)
		self.assertRaises(ValueError, clf.fit, self.KLtr, self.Yte)
		y = clf.predict(self.KLte)
		self.assertEqual(len(self.Yte), len(y))
		clf = clf.fit(self.KLtr_g, self.Ytr)
		self.assertTrue(matNear(w, clf.solution.weights))
		self.assertTrue(matNear(ker, clf.solution.ker_matrix))
		self.assertTrue(matNear(y, clf.predict(self.KLte_g)))
		clf2 = clf.__class__(**clf.get_params())
		clf2.set_params(**clf.get_params())
		err_clf = clf.__class__(**clf.get_params())
		self.assertRaises(NotFittedError, err_clf.predict, self.KLte)
		self.assertRaises(NotFittedError, err_clf.decision_function, self.KLte)


class TestAverageMKL(TestMKL):

	def test_AverageMKL(self):
		self.base_evaluation(algorithms.AverageMKL())
		self.base_evaluation(algorithms.AverageMKL(learner=SVC(C=10)))
		self.base_evaluation(algorithms.AverageMKL(learner=algorithms.KOMD(lam=1)))


class TestEasyMKL(TestMKL):

	def test_EasyMKL(self):
		self.base_evaluation(algorithms.EasyMKL())
		self.base_evaluation(algorithms.EasyMKL(learner=SVC(C=10)))
		self.base_evaluation(algorithms.EasyMKL(learner=algorithms.KOMD(lam=1)))
		self.base_evaluation(algorithms.EasyMKL(solver='libsvm', learner=SVC(C=10)))

	def test_parameters(self):
		self.assertRaises(ValueError, algorithms.EasyMKL, lam=2)
		self.assertRaises(ValueError, algorithms.EasyMKL, lam=1.01)
		self.assertRaises(ValueError, algorithms.EasyMKL, lam=-0.1)
		self.assertRaises(ValueError, algorithms.EasyMKL, solver=0.1)
		algorithms.EasyMKL(solver='libsvm', lam=0.2)


class TestGRAM(TestMKL):

	def test_GRAM(self):
		self.base_evaluation(algorithms.GRAM(max_iter=10))
		self.base_evaluation(algorithms.GRAM(max_iter=10, learner=SVC(C=10)))
		self.base_evaluation(algorithms.GRAM(max_iter=10, learner=algorithms.KOMD(lam=1)))

	def test_callbacks(self):
		earlystop_auc = callbacks.EarlyStopping(
			self.KLte, 
			self.Yte, 
			patience=30, 
			cooldown=2, 
			metric='roc_auc')
		earlystop_acc = callbacks.EarlyStopping(
			self.KLte, 
			self.Yte, 
			patience=30, 
			cooldown=2, 
			metric='accuracy')
		monitor = callbacks.Monitor(metrics=[metrics.radius, metrics.margin, metrics.frobenius])
		cbks = [earlystop_auc, earlystop_acc, monitor]
		clf = algorithms.GRAM(max_iter=60, learning_rate=.001, callbacks=cbks)
		clf = clf.fit(self.KLtr, self.Ytr)
		self.assertEqual(len(monitor.history), 3)
		print (monitor.objective)
		self.assertEqual(len(monitor.objective), 60)
	
	def test_scheduler(self):
		scheduler = ReduceOnWorsening()
		clf = algorithms.GRAM(max_iter=20, learning_rate=.01, scheduler=scheduler)\
			.fit(self.KLtr, self.Ytr)
		scheduler = ReduceOnWorsening(multiplier=.6, min_lr=1e-4)
		clf = algorithms.GRAM(max_iter=20, learning_rate=.01, scheduler=scheduler)\
			.fit(self.KLtr, self.Ytr)
	
	




class TestMulticlass(unittest.TestCase):

	def setUp(self):
		data = load_digits()
		self.Xtr, self.Xte, Ytr, Yte = train_test_split(data.data, data.target, shuffle=True, train_size=.15)
		self.Xtr_numpy = self.Xtr.copy()
		self.Xte_numpy = self.Xte.copy()
		self.Xtr = preprocessing.normalization(self.Xtr)
		self.Xte = preprocessing.normalization(self.Xte)
		self.Ytr = torch.Tensor(Ytr)
		self.Yte = torch.Tensor(Yte)
		self.KLtr = [pairwise_mk.homogeneous_polynomial_kernel(self.Xtr, degree=d) for d in range(1,11)]
		self.KLte = [pairwise_mk.homogeneous_polynomial_kernel(self.Xte, self.Xtr, degree=d) for d in range(1,11)]

	def test_multiclass_ova(self):
		mkl = algorithms.GRAM(multiclass_strategy='ova', learner=SVC(), max_iter=2).fit(self.KLtr, self.Ytr)
		clf = multiclass.OneVsRestMKLClassifier(mkl).fit(self.KLtr, self.Ytr)
		classes = self.Ytr.unique()
		self.assertEqual(len(mkl.solution), len(classes))
		for c in classes.tolist():
			self.assertListEqual(clf.solution[c].weights.tolist(), mkl.solution[c].weights.tolist())

	def test_multiclass_ovo(self):
		mkl = algorithms.EasyMKL(multiclass_strategy='ovo', learner=SVC()).fit(self.KLtr, self.Ytr)
		clf = multiclass.OneVsOneMKLClassifier(mkl).fit(self.KLtr, self.Ytr)
		classes = self.Ytr.unique()
		n = len(classes)
		self.assertEqual(len(mkl.solution), (n*(n-1)/2))
		c1, c2 = classes[:2].tolist()
		self.assertListEqual(clf.solution[(c1,c2)].weights.tolist(), mkl.solution[(c1,c2)].weights.tolist())
		



if __name__ == '__main__':
	unittest.main()


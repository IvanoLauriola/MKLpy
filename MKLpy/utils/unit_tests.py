


import sys
sys.path.append("../..")

import unittest
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics.pairwise import linear_kernel, polynomial_kernel
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from MKLpy import preprocessing
from MKLpy import callbacks
from MKLpy import metrics
from MKLpy.scheduler import ReduceOnWorsening
from MKLpy.metrics import pairwise
from MKLpy import algorithms
from MKLpy.utils.exceptions import SquaredKernelError, InvalidKernelsListError, BinaryProblemError
from MKLpy.utils import misc
from MKLpy.generators import HPK_generator





def matNear(a,b, eps=1e-7):
	return np.isclose(a,b, atol=eps).min()



class TestMetrics(unittest.TestCase):
	def setUp(self):
		self.X = np.array([[0,1], [2,2], [1,1]])
		self.K = self.X.dot(self.X.T)
		self.K1 = np.array([[1,-1,1], [-1,1,-1], [1,-1,1]])
		self.Y = np.array([1,2,1])

	def test_alignment(self):
		self.assertAlmostEqual(metrics.alignment(self.K1, self.K1),1)
		self.assertRaises(SquaredKernelError, metrics.alignment, self.X, self.X)

	def test_alignment_ID(self):
		self.assertLess(metrics.alignment_ID(self.K1), 1)
		self.assertAlmostEqual(metrics.alignment_ID(misc.identity_kernel(self.K1.shape[0])), 1)
		self.assertRaises(SquaredKernelError, metrics.alignment_ID, self.X)

	def test_alignment_yy(self):
		self.assertAlmostEqual(metrics.alignment_yy(self.K1, self.Y), 1)
		self.assertLess(metrics.alignment_yy(self.K, self.Y), 1)

	def test_radius(self):
		self.assertRaises(SquaredKernelError, metrics.radius, self.X)
		self.assertAlmostEqual(metrics.radius(self.K), 5**.5/2, 5)

	def test_margin(self):
		self.assertAlmostEqual(metrics.margin(self.K, self.Y), 2**.5, 5)
		self.assertRaises(SquaredKernelError, metrics.margin, self.X, self.Y)
		self.assertRaises(BinaryProblemError, metrics.margin, self.K, [1,2,3])
		self.assertRaises(ValueError, metrics.margin, self.K, [0,0,1,1,0])

	def test_ratio(self):
		self.assertGreaterEqual(metrics.ratio(self.K, self.Y), 0)
		self.assertRaises(ValueError, metrics.margin, self.K, [0,0,1,1,0])
		self.assertLessEqual(metrics.ratio(self.K, self.Y), 1)
		self.assertRaises(SquaredKernelError, metrics.ratio, self.X, self.Y)
		self.assertRaises(BinaryProblemError, metrics.ratio, self.K, [1,2,3])

	def test_trace(self):
		self.assertEqual(metrics.trace(self.K1), 3)
		self.assertRaises(SquaredKernelError, metrics.trace, self.X)

	def test_frobenius(self):
		self.assertEqual(metrics.frobenius(self.K), 111**.5)
		self.assertEqual(metrics.frobenius(self.K1), 3)
		self.assertRaises(SquaredKernelError, metrics.frobenius, self.X)

	def test_spectral_ratio(self):
		self.assertRaises(SquaredKernelError, metrics.spectral_ratio, self.X, self.Y)
		self.assertEqual(metrics.spectral_ratio(misc.identity_kernel(5), norm=False), 5**.5)
		self.assertEqual(metrics.spectral_ratio(misc.identity_kernel(9), norm=False), 9**.5)
		self.assertEqual(metrics.spectral_ratio(np.ones((5,5)), norm=False), 1)
		self.assertEqual(metrics.spectral_ratio(np.ones((5,5))*4, norm=False), 1)
		self.assertEqual(metrics.spectral_ratio(misc.identity_kernel(5), norm=True), 1)
		self.assertEqual(metrics.spectral_ratio(misc.identity_kernel(9), norm=True), 1)
		self.assertEqual(metrics.spectral_ratio(np.ones((5,5)), norm=True), 0)
		self.assertEqual(metrics.spectral_ratio(np.ones((5,5))*4, norm=True), 0)


class TestPairwise(unittest.TestCase):

	def setUp(self):
		data = load_breast_cancer()
		self.Xtr, self.Xte, self.Ytr, self.Yte = train_test_split(data.data, data.target, shuffle=True, test_size=.2)
		self.Str = ['aaba', 'bac', 'abac', 'waibba', 'aaiicaaac']
		self.Ste = ['aaa','bac','bababbwa']

	def test_HPK_train(self):
		Ktr = self.Xtr.dot(self.Xtr.T)
		self.assertTrue(matNear(Ktr,linear_kernel(self.Xtr)))
		self.assertTrue(matNear(
			pairwise.homogeneous_polynomial_kernel(self.Xtr, degree=4),
			polynomial_kernel(self.Xtr, degree=4, gamma=1, coef0=0)))
		self.assertTrue(matNear(
			pairwise.homogeneous_polynomial_kernel(self.Xtr, degree=5),
			polynomial_kernel(self.Xtr, degree=5, gamma=1, coef0=0)))
		self.assertTrue(matNear(Ktr**3, polynomial_kernel(self.Xtr, degree=3, gamma=1, coef0=0)))
		self.assertTrue(matNear(
			pairwise.homogeneous_polynomial_kernel(self.Xtr, self.Xtr, degree=3),
			polynomial_kernel(self.Xtr, self.Xtr, degree=3, gamma=1, coef0=0)))

	def test_HPK_test(self):
		Ktr = linear_kernel(self.Xtr)
		Kte = self.Xte.dot(self.Xtr.T)
		self.assertTrue(matNear(Kte, 
			pairwise.homogeneous_polynomial_kernel(self.Xte, self.Xtr, degree=1)))
		self.assertTrue(matNear(
			pairwise.homogeneous_polynomial_kernel(self.Xte, self.Xtr, degree=4),
			polynomial_kernel(self.Xte, self.Xtr, degree=4, gamma=1, coef0=0)))

	def test_boolean_kernel(self):
		pass

	def test_spectrum(self):
		Ktr = pairwise.spectrum_kernel(self.Str, p=2)
		self.assertTupleEqual(Ktr.shape, (len(self.Str), len(self.Str)))
		Kte = pairwise.spectrum_kernel(self.Ste, self.Str, p=2)
		self.assertTupleEqual(Kte.shape, (len(self.Ste), len(self.Str)))
		self.assertEqual(Ktr[0,1], 1)
		self.assertEqual(Ktr[0,2], 2)
		self.assertEqual(Kte[0,4], 6)




class TestPreprocessing(unittest.TestCase):

	def setUp(self):
		self.X = np.array([[0,0,1,1], [1,0,1,0], [.5,.5,.5,.5], [0,1,2,3],[-1,0,1,2]])
		self.Y = np.array([0,0,1,1,1])

	def test_normalization(self):
		Xn = preprocessing.normalization(self.X)
		K = Xn.dot(Xn.T)
		self.assertAlmostEqual(K.max(), 1.)
		self.assertAlmostEqual(np.diag(K).min(), 1.)
		self.assertEqual(Xn.shape, (5,4))

	def test_rescale_01(self):
		Xn = preprocessing.rescale_01(self.X)
		self.assertAlmostEqual(Xn.min(), 0)
		self.assertAlmostEqual(Xn.max(), 1)
		self.assertEqual(Xn.shape, (5,4))

	def test_rescale(self):
		Xn = preprocessing.rescale(self.X)
		self.assertAlmostEqual(Xn.min(), -1)
		self.assertAlmostEqual(Xn.max(), 1)
		self.assertEqual(Xn.shape, (5,4))

	def test_centering(self):
		pass

	def test_tracenorm(self):
		K = self.X.dot(self.X.T)
		trace = metrics.trace(K)
		self.assertEqual(trace, 25, K)
		Kt = preprocessing.tracenorm(K)
		self.assertTrue(matNear(Kt*trace/K.shape[0], K))

	def test_kernel_normalization(self):
		pass

	def test_kernel_centering(self):
		pass

	def test_average_binarization(self):
		pass


class TestModelSelection(unittest.TestCase):

	def setUp(self):
		pass

	def test_train_test_split(self):
		pass

	def test_cross_val_score(self):
		pass




class TestMKL(unittest.TestCase):

	def setUp(self):
		data = load_breast_cancer()
		self.Xtr, self.Xte, self.Ytr, self.Yte = train_test_split(data.data, data.target, shuffle=True, train_size=50)
		self.Xtr[:,0] = self.Ytr
		self.Xte[:,0] = self.Yte
		self.Xtr = preprocessing.normalization(self.Xtr)
		self.Xte = preprocessing.normalization(self.Xte)
		self.KLtr = [pairwise.homogeneous_polynomial_kernel(self.Xtr, degree=d) for d in range(1,6)]
		self.KLte = [pairwise.homogeneous_polynomial_kernel(self.Xte, self.Xtr, degree=d) for d in range(1,6)]
		self.KLtr_g = HPK_generator(self.Xtr, degrees=range(1,6))
		self.KLte_g = HPK_generator(self.Xte, self.Xtr, degrees=range(1,6))

	def test_AverageMKL(self):
		self.base_evaluation(algorithms.AverageMKL())
		self.base_evaluation(algorithms.AverageMKL(learner=SVC(C=10)))
		self.base_evaluation(algorithms.AverageMKL(learner=algorithms.KOMD(lam=1)))

	def test_EasyMKL(self):
		self.base_evaluation(algorithms.EasyMKL())
		self.base_evaluation(algorithms.EasyMKL(learner=SVC(C=10)))
		self.base_evaluation(algorithms.EasyMKL(learner=algorithms.KOMD(lam=1)))
		
	def test_GRAM(self):
		self.base_evaluation(algorithms.GRAM(max_iter=10))
		self.base_evaluation(algorithms.GRAM(max_iter=10, learner=SVC(C=10)))
		self.base_evaluation(algorithms.GRAM(max_iter=10, learner=algorithms.KOMD(lam=1)))

	def test_callbacks(self):
		earlystop_auc = callbacks.EarlyStopping(
			self.KLte, 
			self.Yte, 
			patience=5, 
			cooldown=1, 
			metric='auc')
		earlystop_acc = callbacks.EarlyStopping(
			self.KLte, 
			self.Yte, 
			patience=3, 
			cooldown=2, 
			metric='accuracy')
		cbks = [earlystop_auc, earlystop_acc]
		clf = algorithms.GRAM(max_iter=100, learning_rate=.01, callbacks=cbks)
		clf = clf.fit(self.KLtr, self.Ytr)

	def test_scheduler(self):
		scheduler = ReduceOnWorsening()
		clf = algorithms.GRAM(max_iter=10, learning_rate=.01, scheduler=scheduler)\
			.fit(self.KLtr, self.Ytr)
		scheduler = ReduceOnWorsening(multiplier=.6, min_lr=1e-4)
		clf = algorithms.GRAM(max_iter=10, learning_rate=.01, scheduler=scheduler)\
			.fit(self.KLtr, self.Ytr)

	def base_evaluation(self, clf):
		return
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




if __name__ == '__main__':
	unittest.main()
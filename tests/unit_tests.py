


import sys
sys.path.append("..\\MKLpy")

import unittest
import torch
from sklearn.datasets import load_breast_cancer, load_iris, load_digits
from sklearn.metrics import pairwise as pairwise_sk
from sklearn.model_selection import train_test_split
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
from MKLpy.arrange import average





def matNear(a,b, eps=1e-7):
	#return np.isclose(a,b, atol=eps).min()
	a = torch.tensor(a) if type(a) != torch.Tensor else a
	b = torch.tensor(b) if type(b) != torch.Tensor else b
	a = a.type(torch.float64)
	b = b.type(torch.float64)
	return torch.allclose(a, b, atol=eps)



class TestMetrics(unittest.TestCase):
	def setUp(self):
		self.X  = torch.tensor([[0,1], [2,2], [1,1]])
		self.K  = self.X @ self.X.T
		self.K1 = torch.tensor([[1,-1,1], [-1,1,-1], [1,-1,1]])
		self.Y  = torch.tensor([1,2,1])

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
		self.assertRaises(ValueError, metrics.ratio, self.K, [0,0,1,1,0])
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
		self.assertEqual(metrics.spectral_ratio(torch.ones((5,5)), norm=False), 1)
		self.assertEqual(metrics.spectral_ratio(torch.ones((5,5))*4, norm=False), 1)
		self.assertEqual(metrics.spectral_ratio(misc.identity_kernel(5), norm=True), 1)
		self.assertEqual(metrics.spectral_ratio(misc.identity_kernel(9), norm=True), 1)
		self.assertEqual(metrics.spectral_ratio(torch.ones((5,5)), norm=True), 0)
		self.assertEqual(metrics.spectral_ratio(torch.ones((5,5))*4, norm=True), 0)
		o_torch = metrics.spectral_ratio(self.K)
		o_numpy = metrics.spectral_ratio(self.K.numpy())
		self.assertEqual(o_torch, o_numpy)
		self.assertEqual(type(o_torch), float)
		self.assertEqual(type(o_numpy), float)


class TestPairwise(unittest.TestCase):

	def setUp(self):
		data = load_breast_cancer()
		Xtr, Xte = train_test_split(data.data, shuffle=True, test_size=.2)
		self.Xtr = torch.tensor(Xtr)
		self.Xte = torch.tensor(Xte)
		self.Str = ['aaba', 'bac', 'abac', 'waibba', 'aaiicaaac']
		self.Ste = ['aaa','bac','bababbwa']

	def test_HPK_train(self):
		Ktr = self.Xtr @ self.Xtr.T
		self.assertTrue(matNear(Ktr,pairwise_sk.linear_kernel(self.Xtr)))
		self.assertTrue(matNear(
			pairwise_mk.homogeneous_polynomial_kernel(self.Xtr, degree=4),
			pairwise_sk.polynomial_kernel(self.Xtr, degree=4, gamma=1, coef0=0)))
		self.assertTrue(matNear(
			pairwise_mk.homogeneous_polynomial_kernel(self.Xtr, degree=5),
			pairwise_sk.polynomial_kernel(self.Xtr, degree=5, gamma=1, coef0=0)))
		self.assertTrue(matNear(Ktr**3, pairwise_sk.polynomial_kernel(self.Xtr, degree=3, gamma=1, coef0=0)))
		self.assertTrue(matNear(
			pairwise_mk.homogeneous_polynomial_kernel(self.Xtr, self.Xtr, degree=3),
			pairwise_sk.polynomial_kernel(self.Xtr, self.Xtr, degree=3, gamma=1, coef0=0)))

	def test_HPK_test(self):
		Kte = self.Xte @ self.Xtr.T
		self.assertTrue(matNear(Kte, 
			pairwise_mk.homogeneous_polynomial_kernel(self.Xte, self.Xtr, degree=1)))
		self.assertTrue(matNear(
			pairwise_mk.homogeneous_polynomial_kernel(self.Xte, self.Xtr, degree=4),
			pairwise_sk.polynomial_kernel(self.Xte, self.Xtr, degree=4, gamma=1, coef0=0)))

	def test_rbf(self):
		# TODO
		pass

	def test_boolean_kernel(self):
		# TODO
		pass

	def test_spectrum(self):
		Ktr = pairwise_mk.spectrum_kernel(self.Str, p=2)
		self.assertTupleEqual(Ktr.size(), (len(self.Str), len(self.Str)))
		Kte = pairwise_mk.spectrum_kernel(self.Ste, self.Str, p=2)
		self.assertTupleEqual(Kte.size(), (len(self.Ste), len(self.Str)))
		self.assertEqual(Ktr[0,1], 1)
		self.assertEqual(Ktr[0,2], 2)
		self.assertEqual(Kte[0,4], 6)

	def test_otype(self):
		self.assertEqual(type(pairwise_mk.linear_kernel(self.Xtr)), torch.Tensor)
		self.assertEqual(type(pairwise_mk.homogeneous_polynomial_kernel(self.Xtr)), torch.Tensor)
		self.assertEqual(type(pairwise_mk.polynomial_kernel(self.Xtr)), torch.Tensor)
		self.assertEqual(type(pairwise_mk.rbf_kernel(self.Xtr)), torch.Tensor)

	def test_numpy(self):
		Xtr = self.Xtr.numpy()
		self.assertTrue(matNear(
			pairwise_mk.polynomial_kernel(Xtr, degree=4, gamma=0.1, coef0=2),
			pairwise_sk.polynomial_kernel(Xtr, degree=4, gamma=0.1, coef0=2)))
		self.assertTrue(matNear(
			pairwise_mk.linear_kernel(Xtr),
			pairwise_sk.linear_kernel(Xtr)))




class TestPreprocessing(unittest.TestCase):

	def setUp(self):
		self.X = torch.tensor([[0,0,1,1], [1,0,1,0], [.5,.5,1,.5], [0,2,1,3],[-1,0,1,2]])
		self.Xnumpy = self.X.numpy()

	def test_normalization(self):
		Xn = preprocessing.normalization(self.X)
		K = Xn @ Xn.T
		self.assertAlmostEqual(K.max().item(), 1., places=6)
		self.assertAlmostEqual(K.diag().min().item(), 1., places=6)
		self.assertEqual(Xn.shape, (5,4))
		o_torch = preprocessing.normalization(self.X)
		o_numpy = preprocessing.normalization(self.Xnumpy)
		self.assertTrue(matNear(o_torch, o_numpy))
		self.assertEqual(type(o_torch), torch.Tensor)
		self.assertEqual(type(o_numpy), torch.Tensor)

	def test_rescale_01(self):
		Xn = preprocessing.rescale_01(self.X)
		self.assertAlmostEqual(Xn.min().item(), 0)
		self.assertAlmostEqual(Xn.max().item(), 1)
		self.assertEqual(Xn.shape, (5,4))
		self.assertEqual(Xn[2,0], 0.75)
		self.assertEqual(Xn[2,2], 0)
		o_torch = preprocessing.rescale_01(self.X)
		o_numpy = preprocessing.rescale_01(self.Xnumpy)
		self.assertTrue(matNear(o_torch, o_numpy))
		self.assertEqual(type(o_torch), torch.Tensor)
		self.assertEqual(type(o_numpy), torch.Tensor)

	def test_rescale(self):
		Xn = preprocessing.rescale(self.X)
		self.assertAlmostEqual(Xn.min().item(), -1)
		self.assertAlmostEqual(Xn.max().item(), +1)
		self.assertEqual(Xn.shape, (5,4))
		self.assertEqual(Xn[2,0], 0.5)
		self.assertEqual(Xn[2,2], 0)
		o_torch = preprocessing.rescale(self.X)
		o_numpy = preprocessing.rescale(self.Xnumpy)
		self.assertTrue(matNear(o_torch, o_numpy))
		self.assertEqual(type(o_torch), torch.Tensor)
		self.assertEqual(type(o_numpy), torch.Tensor)

	def test_centering(self):
		# TODO
		pass

	def test_tracenorm(self):
		K = self.X @ self.X.T
		trace = metrics.trace(K)
		self.assertEqual(trace, 25.75, K)
		Kt = preprocessing.tracenorm(K)
		self.assertTrue(matNear(Kt*trace/K.shape[0], K))
		o_numpy = metrics.trace(K.numpy())
		self.assertEqual(trace, o_numpy)

	def test_kernel_normalization(self):
		K = self.X @ self.X.T
		Kn_torch = preprocessing.kernel_normalization(K)
		Kn_numpy = preprocessing.kernel_normalization(K.numpy())
		self.assertAlmostEqual(Kn_torch.max().item(), 1., places=6)
		self.assertAlmostEqual(Kn_torch.diag().min().item(), 1., places=6)
		self.assertEqual(Kn_torch.shape, (5,5))
		self.assertTrue(matNear(Kn_torch, Kn_numpy))
		self.assertEqual(type(Kn_torch), torch.Tensor)
		self.assertEqual(type(Kn_numpy), torch.Tensor)
		linear = pairwise_mk.linear_kernel(preprocessing.normalization(self.X))
		self.assertTrue(matNear(Kn_torch, linear, eps=1e-7))

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
		self.Xtr = preprocessing.normalization(self.Xtr)
		self.Xte = preprocessing.normalization(self.Xte)
		self.KLtr = [pairwise_mk.homogeneous_polynomial_kernel(self.Xtr, degree=d) for d in range(1,6)]
		self.KLte = [pairwise_mk.homogeneous_polynomial_kernel(self.Xte, self.Xtr, degree=d) for d in range(1,6)]
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


class TestGenerators(unittest.TestCase):

	def setUp(self):
		data = load_breast_cancer()
		self.Xtr, self.Xte, self.Ytr, self.Yte = train_test_split(data.data, data.target, shuffle=True, train_size=50)

	def test_hpk(self):
		KLtr = [pairwise_mk.homogeneous_polynomial_kernel(self.Xtr, degree=d) for d in range(1,6)]
		KLte = [pairwise_mk.homogeneous_polynomial_kernel(self.Xte, self.Xtr, degree=d) for d in range(1,6)]
		KLtr_g = HPK_generator(self.Xtr, degrees=range(1,6))
		KLte_g = HPK_generator(self.Xte, self.Xtr, degrees=range(1,6), cache=False)
		self.assertTrue(matNear(average(KLtr), average(KLtr_g)))
		self.assertTrue(matNear(average(KLte), average(KLte_g)))
		self.assertTrue(matNear(KLtr[1], KLtr_g[1]))
		self.assertTrue(matNear(KLte[2], KLte_g[2]))

	def test_rbf(self):
		gammavals = [0.001, 0.01, 0.1, 1.]
		KLtr = [pairwise_mk.rbf_kernel(self.Xtr, gamma=d) for d in gammavals]
		KLte = [pairwise_mk.rbf_kernel(self.Xte, self.Xtr, gamma=d) for d in gammavals]
		KLtr_g = RBF_generator(self.Xtr, gamma=gammavals, cache=False)
		KLte_g = RBF_generator(self.Xte, self.Xtr, gamma=gammavals)
		self.assertTrue(matNear(average(KLtr), average(KLtr_g)))
		self.assertTrue(matNear(average(KLte), average(KLte_g)))
		self.assertTrue(matNear(KLtr[1], KLtr_g[1]))
		self.assertTrue(matNear(KLte[2], KLte_g[2]))

	def test_lambda(self):
		funcs = [pairwise_mk.linear_kernel, lambda X,Z : (X @ Z.T)**2]
		KLtr = [pairwise_mk.linear_kernel(self.Xtr), pairwise_mk.homogeneous_polynomial_kernel(self.Xtr)]
		KLte = [pairwise_mk.linear_kernel(self.Xte, self.Xtr), pairwise_mk.homogeneous_polynomial_kernel(self.Xte, self.Xtr)]
		KLtr_g = Lambda_generator(self.Xtr, kernels=funcs)
		KLte_g = Lambda_generator(self.Xte, self.Xtr, kernels=funcs)
		self.assertTrue(matNear(average(KLtr), average(KLtr_g)))
		self.assertTrue(matNear(average(KLte), average(KLte_g)))
		self.assertTrue(matNear(KLtr[1], KLtr_g[1]))
		self.assertTrue(matNear(KLte[0], KLte_g[0]))


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
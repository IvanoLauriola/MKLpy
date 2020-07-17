


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




class TestKernel(unittest.TestCase):
	def setUp(self):
		data = load_breast_cancer()
		Xtr, Xte = train_test_split(data.data, shuffle=True, test_size=.4, random_state=42)
		self.Xtr = torch.tensor(Xtr)
		self.Xte = torch.tensor(Xte)


class TestLinear(TestKernel):

	def test_comptuation(self):
		self.assertTrue(matNear(
			pairwise_sk.linear_kernel(self.Xtr), 
			pairwise_mk.linear_kernel(self.Xtr)))
		self.assertTrue(matNear(
			pairwise_sk.linear_kernel(self.Xte, self.Xtr), 
			pairwise_mk.linear_kernel(self.Xte, self.Xtr)))

	def test_type(self):
		Kte_torch = pairwise_mk.linear_kernel(self.Xte, self.Xtr)
		self.assertEqual(type(pairwise_mk.linear_kernel(self.Xtr)), torch.Tensor)
		self.assertEqual(type(pairwise_mk.linear_kernel(self.Xtr.tolist())), torch.Tensor)
		self.assertEqual(type(Kte_torch), torch.Tensor)
		self.assertTrue(matNear(Kte_torch,
			pairwise_mk.linear_kernel(self.Xte.numpy(), self.Xtr.numpy())))
		self.assertTrue(matNear(Kte_torch,
			pairwise_mk.linear_kernel(self.Xte.tolist(), self.Xtr.tolist())))

	def test_raise(self):
		self.assertRaises(ValueError, pairwise_mk.linear_kernel, self.Xtr[0])
		self.assertRaises(ValueError, pairwise_mk.linear_kernel, [self.Xtr]*2)
		self.assertRaises(ValueError, pairwise_mk.linear_kernel, self.Xtr, self.Xte.T)

	def test_shape(self):
		self.assertTupleEqual(pairwise_mk.linear_kernel(self.Xte, self.Xtr).numpy().shape, (self.Xte.size()[0], self.Xtr.size()[0]) )
		self.assertTupleEqual(pairwise_mk.linear_kernel(self.Xtr).numpy().shape, (self.Xtr.size()[0], self.Xtr.size()[0]) )


class TestHPK(TestKernel):

	def test_comptuation(self):
		Ktr = pairwise_mk.homogeneous_polynomial_kernel(self.Xtr, degree=1)
		self.assertTrue(matNear(Ktr, pairwise_sk.linear_kernel(self.Xtr)))
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
		self.assertTrue(matNear(
			pairwise_mk.homogeneous_polynomial_kernel(self.Xte, self.Xtr, degree=1), 
			pairwise_mk.homogeneous_polynomial_kernel(self.Xte, self.Xtr, degree=1)))
		self.assertTrue(matNear(
			pairwise_mk.homogeneous_polynomial_kernel(self.Xte, self.Xtr, degree=4),
			pairwise_sk.polynomial_kernel(self.Xte, self.Xtr, degree=4, gamma=1, coef0=0)))


	def test_type(self):
		Kte_torch = pairwise_mk.homogeneous_polynomial_kernel(self.Xte, self.Xtr, degree=4)
		self.assertEqual(type(pairwise_mk.homogeneous_polynomial_kernel(self.Xtr)), torch.Tensor)
		self.assertEqual(type(pairwise_mk.homogeneous_polynomial_kernel(self.Xtr.tolist())), torch.Tensor)
		self.assertEqual(type(Kte_torch), torch.Tensor)
		self.assertTrue(matNear(Kte_torch,
			pairwise_mk.homogeneous_polynomial_kernel(self.Xte.numpy(), self.Xtr.numpy(), degree=4)))
		self.assertTrue(matNear(Kte_torch,
			pairwise_mk.homogeneous_polynomial_kernel(self.Xte.tolist(), self.Xtr.tolist(), degree=4)))

	def test_raise(self):
		self.assertRaises(ValueError, pairwise_mk.homogeneous_polynomial_kernel, self.Xtr[0])
		self.assertRaises(ValueError, pairwise_mk.homogeneous_polynomial_kernel, [self.Xtr]*2)
		self.assertRaises(ValueError, pairwise_mk.homogeneous_polynomial_kernel, self.Xtr, self.Xte.T)

	def test_shape(self):
		self.assertTupleEqual(pairwise_mk.homogeneous_polynomial_kernel(self.Xte, self.Xtr).numpy().shape, (self.Xte.size()[0], self.Xtr.size()[0]) )
		self.assertTupleEqual(pairwise_mk.homogeneous_polynomial_kernel(self.Xtr).numpy().shape, (self.Xtr.size()[0], self.Xtr.size()[0]) )


class TestPolynomial(TestKernel):

	def test_comptuation(self):
		self.assertTrue(matNear(
			pairwise_mk.polynomial_kernel(self.Xtr, degree=1, coef0=0, gamma=1),
			pairwise_sk.linear_kernel(self.Xtr)))
		self.assertTrue(matNear(
			pairwise_mk.polynomial_kernel(self.Xtr, degree=4, gamma=1.5, coef0=2),
			pairwise_sk.polynomial_kernel(self.Xtr, degree=4, gamma=1.5, coef0=2)))
		self.assertTrue(matNear(
			pairwise_mk.polynomial_kernel(self.Xtr, self.Xtr, degree=3, gamma=2, coef0=0),
			pairwise_sk.polynomial_kernel(self.Xtr, self.Xtr, degree=3, gamma=2, coef0=0)))
		self.assertTrue(matNear(
			pairwise_mk.polynomial_kernel(self.Xte, self.Xtr, degree=1),
			pairwise_mk.polynomial_kernel(self.Xte, self.Xtr, degree=1)))
		self.assertTrue(matNear(
			pairwise_mk.polynomial_kernel(self.Xte, self.Xtr, degree=3, gamma=1.6, coef0=.3),
			pairwise_sk.polynomial_kernel(self.Xte, self.Xtr, degree=3, gamma=1.6, coef0=.3)))


	def test_type(self):
		Kte_torch = pairwise_mk.polynomial_kernel(self.Xte, self.Xtr, degree=4)
		self.assertEqual(type(pairwise_mk.polynomial_kernel(self.Xtr)), torch.Tensor)
		self.assertEqual(type(pairwise_mk.polynomial_kernel(self.Xtr.tolist())), torch.Tensor)
		self.assertEqual(type(Kte_torch), torch.Tensor)
		self.assertTrue(matNear(Kte_torch,
			pairwise_mk.polynomial_kernel(self.Xte.numpy(), self.Xtr.numpy(), degree=4)))
		self.assertTrue(matNear(Kte_torch,
			pairwise_mk.polynomial_kernel(self.Xte.tolist(), self.Xtr.tolist(), degree=4)))

	def test_raise(self):
		self.assertRaises(ValueError, pairwise_mk.polynomial_kernel, self.Xtr[0])
		self.assertRaises(ValueError, pairwise_mk.polynomial_kernel, [self.Xtr]*2)
		self.assertRaises(ValueError, pairwise_mk.polynomial_kernel, self.Xtr, self.Xte.T)

	def test_shape(self):
		self.assertTupleEqual(pairwise_mk.polynomial_kernel(self.Xte, self.Xtr).numpy().shape, (self.Xte.size()[0], self.Xtr.size()[0]) )
		self.assertTupleEqual(pairwise_mk.polynomial_kernel(self.Xtr).numpy().shape, (self.Xtr.size()[0], self.Xtr.size()[0]) )


class TestRBF(TestKernel):

	def test_comptuation(self):
		self.assertTrue(matNear(
			pairwise_sk.rbf_kernel(self.Xtr, gamma=.01), 
			pairwise_mk.rbf_kernel(self.Xtr, gamma=.01)))
		self.assertTrue(matNear(
			pairwise_sk.rbf_kernel(self.Xte, self.Xtr, gamma=.1), 
			pairwise_mk.rbf_kernel(self.Xte, self.Xtr, gamma=.1)))
		self.assertTrue(matNear(
			pairwise_sk.rbf_kernel(self.Xte, self.Xtr, gamma=.001), 
			pairwise_mk.rbf_kernel(self.Xte, self.Xtr, gamma=.001)))

	def test_type(self):
		Kte_torch = pairwise_mk.rbf_kernel(self.Xte, self.Xtr, gamma=.01)
		self.assertEqual(type(pairwise_mk.rbf_kernel(self.Xtr, gamma=.2)), torch.Tensor)
		self.assertEqual(type(pairwise_mk.rbf_kernel(self.Xtr.tolist(), gamma=.2)), torch.Tensor)
		self.assertEqual(type(Kte_torch), torch.Tensor)
		self.assertTrue(matNear(Kte_torch,
			pairwise_mk.rbf_kernel(self.Xte.numpy(), self.Xtr.numpy(), gamma=.01)))
		self.assertTrue(matNear(Kte_torch,
			pairwise_mk.rbf_kernel(self.Xte.tolist(), self.Xtr.tolist(), gamma=.01), eps=1e-6))

	def test_raise(self):
		self.assertRaises(ValueError, pairwise_mk.rbf_kernel, self.Xtr[0])
		self.assertRaises(ValueError, pairwise_mk.rbf_kernel, [self.Xtr]*2)
		self.assertRaises(ValueError, pairwise_mk.rbf_kernel, self.Xtr, self.Xte.T)

	def test_shape(self):
		self.assertTupleEqual(pairwise_mk.rbf_kernel(self.Xte, self.Xtr).numpy().shape, (self.Xte.size()[0], self.Xtr.size()[0]) )
		self.assertTupleEqual(pairwise_mk.rbf_kernel(self.Xtr).numpy().shape, (self.Xtr.size()[0], self.Xtr.size()[0]) )








class TestPairwise(unittest.TestCase):
	#to be moved in specific classes

	def setUp(self):
		data = load_breast_cancer()
		Xtr, Xte = train_test_split(data.data, shuffle=True, test_size=.2)
		self.Xtr = torch.tensor(Xtr)
		self.Xte = torch.tensor(Xte)
		self.Str = ['aaba', 'bac', 'abac', 'waibba', 'aaiicaaac']
		self.Ste = ['aaa','bac','bababbwa']



	def test_boolean_kernel(self):
		X = torch.tensor([[1,0,1,1],[0,0,1,1],[1,0,0,0]]).numpy()
		Kmc = pairwise_mk.monotone_conjunctive_kernel(X, c=2)
		Kmd = pairwise_mk.monotone_disjunctive_kernel(X, d=2)
		Kmdnf = pairwise_mk.monotone_dnf_kernel(X, d=2, c=3)
		self.assertEqual(Kmc[0,0], 3)
		self.assertEqual(Kmc[1,1], 1)
		self.assertEqual(Kmc[2,2], 0)
		self.assertEqual(Kmc[0,1], 1)

	def test_spectrum(self):
		Ktr = pairwise_mk.spectrum_kernel(self.Str, p=2)
		self.assertTupleEqual(Ktr.size(), (len(self.Str), len(self.Str)))
		Kte = pairwise_mk.spectrum_kernel(self.Ste, self.Str, p=2)
		self.assertTupleEqual(Kte.size(), (len(self.Ste), len(self.Str)))
		self.assertEqual(Ktr[0,1], 1)
		self.assertEqual(Ktr[0,2], 2)
		self.assertEqual(Kte[0,4], 6)




##### TEST kernels generators

class TestGenerators(unittest.TestCase):

	def setUp(self):
		data = load_breast_cancer()
		Xtr, Xte = train_test_split(data.data, shuffle=True, train_size=.2)
		self.Xtr = torch.tensor(Xtr)
		self.Xte = torch.tensor(Xte)

	def _check_lists(self, KL1, KL2):
		self.assertTrue(matNear(average(KL1), average(KL2)))
		self.assertTrue(matNear(KL1[2], KL2[2]))
		self.assertEqual(len(KL1), len(KL2))


class TestGenHPK(TestGenerators):

	def setUp(self):
		super().setUp()
		self.KLtr = [pairwise_mk.homogeneous_polynomial_kernel(self.Xtr, degree=d) for d in range(1,6)]
		self.KLte = [pairwise_mk.homogeneous_polynomial_kernel(self.Xte, self.Xtr, degree=d) for d in range(1,6)]

	def test_computation(self):
		KLtr_g = HPK_generator(self.Xtr, degrees=range(1,6))
		KLte_g = HPK_generator(self.Xte, self.Xtr, degrees=range(1,6))
		self._check_lists(self.KLtr, KLtr_g)
		self._check_lists(self.KLte, KLte_g)
		KLtr_g = HPK_generator(self.Xtr, self.Xtr, degrees=range(1,6))
		self._check_lists(self.KLtr, KLtr_g)

	def test_caching(self):
		KLtr_g = HPK_generator(self.Xtr, degrees=range(1,6))
		KLte_g = HPK_generator(self.Xte, self.Xtr, degrees=range(1,6))
		KLtr_g_c = HPK_generator(self.Xtr, degrees=range(1,6), cache=True)
		KLte_g_c = HPK_generator(self.Xte, self.Xtr, degrees=range(1,6), cache=True)
		self._check_lists(KLtr_g_c, KLtr_g)
		self._check_lists(KLte_g_c, KLte_g)

	def test_identity(self):
		KLtr_g_i  = HPK_generator(self.Xtr, degrees=range(1,6), include_identity=True)
		KLte_g_i = HPK_generator(self.Xte, self.Xtr, degrees=range(1,6), include_identity=True, cache=True)
		KLte_g_i_c = HPK_generator(self.Xte, self.Xtr, degrees=range(1,6), include_identity=True, cache=False)
		I = misc.identity_kernel(len(self.Xtr))
		Z = torch.zeros(len(self.Xte), len(self.Xtr))
		self._check_lists(KLtr_g_i, self.KLtr + [I])
		self._check_lists(KLte_g_i, self.KLte + [Z])
		self._check_lists(KLte_g_i_c, self.KLte + [Z])
		

class TestGenRBF(TestGenerators):

	def setUp(self):
		super().setUp()
		self.gammavals = [0.001, 0.01, 0.1, 1]
		self.KLtr = [pairwise_mk.rbf_kernel(self.Xtr, gamma=g) for g in self.gammavals]
		self.KLte = [pairwise_mk.rbf_kernel(self.Xte, self.Xtr, gamma=g) for g in self.gammavals]

	def test_computation(self):
		KLtr_g = RBF_generator(self.Xtr, gamma=self.gammavals)
		KLte_g = RBF_generator(self.Xte, self.Xtr, gamma=self.gammavals)
		self._check_lists(self.KLtr, KLtr_g)
		self._check_lists(self.KLte, KLte_g)
		KLtr_g = RBF_generator(self.Xtr, self.Xtr, gamma=self.gammavals)
		self._check_lists(self.KLtr, KLtr_g)

	def test_caching(self):
		KLtr_g = RBF_generator(self.Xtr, gamma=self.gammavals)
		KLte_g = RBF_generator(self.Xte, self.Xtr, gamma=self.gammavals)
		KLtr_g_c = RBF_generator(self.Xtr, gamma=self.gammavals, cache=True)
		KLte_g_c = RBF_generator(self.Xte, self.Xtr, gamma=self.gammavals, cache=True)
		self._check_lists(KLtr_g_c, KLtr_g)
		self._check_lists(KLte_g_c, KLte_g)

	def test_identity(self):
		KLtr_g_i  = RBF_generator(self.Xtr, gamma=self.gammavals, include_identity=True)
		KLte_g_i = RBF_generator(self.Xte, self.Xtr, gamma=self.gammavals, include_identity=True, cache=True)
		KLte_g_i_c = RBF_generator(self.Xte, self.Xtr, gamma=self.gammavals, include_identity=True, cache=False)
		I = misc.identity_kernel(len(self.Xtr))
		Z = torch.zeros(len(self.Xte), len(self.Xtr))
		self._check_lists(KLtr_g_i, self.KLtr + [I])
		self._check_lists(KLte_g_i, self.KLte + [Z])
		self._check_lists(KLte_g_i_c, self.KLte + [Z])


class TestGenLambda(TestGenerators):

	def setUp(self):
		super().setUp()
		self.funcs = [
			pairwise_mk.linear_kernel, 
			lambda X,Z : (X @ Z.T)**2, 
			pairwise_mk.polynomial_kernel,
			lambda X,Z : pairwise_mk.polynomial_kernel(X, Z, degree=4),
		]
		self.KLtr = [f(self.Xtr, self.Xtr) for f in self.funcs]
		self.KLte = [
			pairwise_mk.linear_kernel(self.Xte, self.Xtr),
			pairwise_mk.homogeneous_polynomial_kernel(self.Xte, self.Xtr, degree=2),
			pairwise_mk.polynomial_kernel(self.Xte, self.Xtr),
			pairwise_mk.polynomial_kernel(self.Xte, self.Xtr, degree=4),
		]

	def test_computation(self):
		KLtr_g = Lambda_generator(self.Xtr, kernels=self.funcs)
		KLte_g = Lambda_generator(self.Xte, self.Xtr, kernels=self.funcs)
		self._check_lists(self.KLtr, KLtr_g)
		self._check_lists(self.KLte, KLte_g)
		KLtr_g = Lambda_generator(self.Xtr, self.Xtr, kernels=self.funcs)
		self._check_lists(self.KLtr, KLtr_g)

	def test_caching(self):
		self.assertRaises(TypeError, Lambda_generator, self.Xtr, kernels=self.funcs, cache=True)
		self.assertRaises(TypeError, Lambda_generator, self.Xte, self.Xtr, kernels=self.funcs, cache=True)

	def test_identity(self):
		KLtr_g_i  = Lambda_generator(self.Xtr, kernels=self.funcs, include_identity=True)
		KLte_g_i = Lambda_generator(self.Xte, self.Xtr, kernels=self.funcs, include_identity=True)
		I = misc.identity_kernel(len(self.Xtr))
		Z = torch.zeros(len(self.Xte), len(self.Xtr))
		self._check_lists(KLtr_g_i, self.KLtr + [I])
		self._check_lists(KLte_g_i, self.KLte + [Z])



class TestGenMultiview(TestGenerators):

	def setUp(self):
		super().setUp()
		self.XLtr = [self.Xtr + i for i in range(5)]
		self.XLte = [self.Xte + i for i in range(5)]
		self.kf   = lambda _X, _Z : pairwise_mk.homogeneous_polynomial_kernel(_X, _Z, degree=2)
		self.KLtr = [self.kf(X, X)  for X in self.XLtr]
		self.KLte = [self.kf(Xt, X) for Xt,X in zip(self.XLte, self.XLtr)]

	def test_computation(self):
		KLtr_g = Multiview_generator(self.XLtr, kernel=self.kf)
		KLte_g = Multiview_generator(self.XLte, self.XLtr, kernel=self.kf)
		self._check_lists(self.KLtr, KLtr_g)
		self._check_lists(self.KLte, KLte_g)
		KLtr_g = Multiview_generator(self.XLtr, self.XLtr, kernel=self.kf)
		self._check_lists(self.KLtr, KLtr_g)

	def test_caching(self):
		self.assertRaises(TypeError, Lambda_generator, self.XLtr, kernels=self.kf, cache=True)
		self.assertRaises(TypeError, Lambda_generator, self.XLtr, kernels=self.kf, cache=True)

	def test_identity(self):
		KLtr_g_i  = Multiview_generator(self.XLtr, kernel=self.kf, include_identity=True)
		KLte_g_i = Multiview_generator(self.XLte, self.XLtr, kernel=self.kf, include_identity=True)
		I = misc.identity_kernel(len(self.Xtr))
		Z = torch.zeros(len(self.Xte), len(self.Xtr))
		self._check_lists(KLtr_g_i, self.KLtr + [I])
		self._check_lists(KLte_g_i, self.KLte + [Z])

	def test_shape(self):
		self.assertRaises(ValueError, Multiview_generator, self.Xtr, kernel=self.kf, include_identity=True)
		self.assertRaises(ValueError, Multiview_generator, self.Xte, self.Xtr, kernel=self.kf, include_identity=True)


if __name__ == '__main__':
	unittest.main()





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






class TestModelSelection(unittest.TestCase):

	def setUp(self):
		X = torch.tensor(
			[[2,2,2,3], [0,0,1,1], [1,0,1,0], [.5,.5,1,.5], [0,2,1,3],[-1,0,1,2]]*2)
		self.KL = [(X @ X.T)**d for d in range(1,6)]
		self.Y = torch.tensor([1,0,0,0,1,1]*2)

	def test_train_test_split(self):
		Ktr, Kte, Ytr, Yte = tts(self.KL, self.Y, test_size=2)
		self.assertEqual(Ktr[0].size(), (10,10))
		self.assertEqual(Kte[0].size(), (2,10))
		self.assertEqual(type(Ktr[0]), torch.Tensor)
		self.assertEqual(type(Ytr), torch.Tensor)

	def test_cross_val_score(self):
		mkl = algorithms.AverageMKL()
		scores = cross_val_score(self.KL, self.Y, mkl)
		self.assertEqual(len(scores), 3)
		self.assertEqual(len(cross_val_score(self.KL, self.Y, mkl, n_folds=5)), 5)
		self.assertRaises(ValueError, cross_val_score, self.KL, self.Y, mkl, scoring='pippo franco')
		loo = LeaveOneOut()
		scores = cross_val_score(self.KL, self.Y, mkl, cv=loo, scoring='accuracy')
		self.assertEqual(len(scores), len(self.Y))





if __name__ == '__main__':
	unittest.main()


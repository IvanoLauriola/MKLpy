


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
		binarizer = AverageBinarizer().fit(self.X)
		Xt = binarizer.transform(self.X)
		self.assertEqual(self.X.size(), Xt.size())
		self.assertTrue(not binarizer.get_params()['duplicate'])
		binarizer.set_params(duplicate=True)
		self.assertTrue(binarizer.get_params()['duplicate'])
		Xtd = AverageBinarizer(duplicate=True).fit_transform(self.X)
		self.assertEqual((5,8), Xtd.size())
		self.assertListEqual(Xt.numpy()[0].tolist(), Xt.numpy()[0].tolist())
		self.assertListEqual(Xt.numpy()[2].tolist(), [1,0,0,0])
		self.assertListEqual(Xt.numpy()[4].tolist(), [0,0,0,1])




if __name__ == '__main__':
	unittest.main()


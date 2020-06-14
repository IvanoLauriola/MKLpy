


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






class TestOperations(unittest.TestCase):

	def setUp(self):
		data = load_breast_cancer()
		self.Xtr, self.Xte, self.Ytr, self.Yte = train_test_split(data.data, data.target, shuffle=True, train_size=50)
		self.KLtr = [self.Xtr @self.Xtr.T **i for i in range(1,5)]
		self.KLte = [self.Xte @self.Xtr.T **i for i in range(1,5)]
		self.w = [0.2, 0.5, 0.1, 0.2]

	def test_multiply(self):
		KLtr_mul, KLte_mul = multiplication(self.KLtr), multiplication(self.KLte)
	
	def test_average(self):
		KLtr_ave, KLte_ave = average(self.KLtr), average(self.KLte)

	def test_summation(self):
		KLtr_sum, KLte_sum = summation(self.KLtr), summation(self.KLte)


if __name__ == '__main__':
	unittest.main()


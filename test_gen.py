

class Gen:

	def __init__(self,l):
		self.max = l

	def __iter__(self):
		self.idx = 0
		return self

	def __next__(self):
		if self.idx >= self.max:
			raise StopIteration
		self.idx += 1
		return self.__getitem__(self.idx)

	def __getitem__(self, r):
		return 10 + r

	def __len__(self):
		return self.max


from MKLpy.algorithms import AverageMKL
import numpy as np 
from MKLpy.generators import HPK_generator
from sklearn.datasets import load_svmlight_file
from MKLpy.preprocessing import normalization
X,Y = load_svmlight_file('C:/Users/Ivan/Dropbox/RICERCA/FIM/svmlight/mushroom.svmlight')
X = normalization(X.toarray())
print (X.shape)

from MKLpy.arrange import average

#KL = [X.dot(X.T)**d for d in range(1, 11)]
KL = HPK_generator(X, Z=X, degrees=range(1,11))

#K = average(KL)

from MKLpy.algorithms import EasyMKL
from sklearn.svm import SVC
clf = EasyMKL(SVC()).fit(KL,Y)
#Ksum = np.zeros((n,n))
#for K in KL:
#	Ksum += K

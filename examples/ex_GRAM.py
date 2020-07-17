'''

This is a snippet of code showing how to train a MKL algorithm

Author: Ivano Lauriola, ivano.lauriola@phd.unipd.it

'''

#load data
from sklearn.datasets import load_breast_cancer as load
ds = load()
X,Y = ds.data, ds.target

from MKLpy.preprocessing import normalization
X = normalization(X)


from sklearn.model_selection import train_test_split
Xtr,Xte,Ytr,Yte = train_test_split(X,Y, test_size=.5, random_state=42)


from MKLpy.metrics import pairwise
from MKLpy.utils.misc import identity_kernel
import torch

#making 20 homogeneous polynomial kernels.
#I suggest to add the identity kernel in order to make the GRAM initial solution easily separable
#if the initial sol is not separable, GRAM may not work well
KLtr = [pairwise.homogeneous_polynomial_kernel(Xtr, degree=d) for d in range(1,11)] + [identity_kernel(len(Ytr))]
KLte = [pairwise.homogeneous_polynomial_kernel(Xte,Xtr, degree=d) for d in range(1,11)]
KLte.append(torch.zeros(KLte[0].size()))


from MKLpy.algorithms import GRAM
from MKLpy.scheduler import ReduceOnWorsening
from MKLpy.callbacks import EarlyStopping

earlystop = EarlyStopping(
	KLte, 
	Yte, 
	patience=100,
	cooldown=1, 
	metric='roc_auc',
)
scheduler = ReduceOnWorsening()

clf = GRAM(
	max_iter=100, 
	learning_rate=.1, 
	callbacks=[earlystop], 
	scheduler=scheduler).fit(KLtr, Ytr)


'''

This is a snippet of code containing basic functions of the package, including:
- how to preprocess input data
- how to compute lists of kernels (HPKs in the example)
- some useful metrics on kernels (radius, margin, spectral ratio, etc...)

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


#evaluate kernels in terms of margin, radius etc...
print ('evaluating metrics...', end='')
from MKLpy.metrics import margin, radius, ratio, trace, frobenius
from MKLpy.preprocessing import kernel_normalization
deg = 5
K = KLtr[deg]					#the HPK with degree 5
K = kernel_normalization(K)		#normalize the kernel K (useless in the case of HPK computed on normalized data)
score_margin = margin(K,Ytr)	#the distance between the positive and negative classes in the kernel space
score_radius = radius(K)		#the radius of the Einimum Enclosing Ball containing data in the kernel space
score_ratio  = ratio (K,Ytr)	#the radius/margin ratio defined as (radius**2/margin**2)/n_examples
#the ratio can be also computed as score_radius**2/score_margin**2/len(Ytr)
score_trace  = trace (K)		#the trace of the kernel matrix
score_froben = frobenius(K)		#the Frobenius norm of a kernel matrix
print ('done')
print ('results of the %d-degree HP kernel:' % deg)
print ('margin: %.4f, radius: %.4f, radiu-margin ratio: %.4f,' % (score_margin, score_radius, score_ratio))
print ('trace: %.4f, frobenius norm: %.4f' % (score_trace, score_froben))


#evaluate the empirical complexity of the kernel matrix, i.e. the Spectral Ratio
# Michele Donini, Fabio Aiolli: "Learning deep kernels in the space of dot-product polynomials". Machine Learning (2017)
# Ivano Lauriola, Mirko Polato, Fabio Aiolli: "The Minimum Effort Maximum Output principle applied to Multiple Kernel Learning". ESANN (2018)
print ('computing Spectral Ratio...', end='')
from MKLpy.metrics import spectral_ratio
SR = spectral_ratio(K, norm=True)
print ('%.4f' % SR)

# Examples

This page contains several base examples to use MKLpy and its features.



### Loading data


Datasets are numpy matrices, and they can be loaded by means of scikit-learn routines

```python
from sklearn.datasets import load_breast_cancer
ds = load_breast_cancer()
X,Y = ds.data, ds.target
```

```python
from sklearn.datasets import load_svmlight_file
X,Y = load_svmlight_file(...)
X = X.toarray()	#be sure that your matrix is not sparse!
```

- - -

### Preprocessing

```python
from MKLpy.preprocessing import normalization, rescale_01
X = rescale_01(X)		#feature scaling in [0,1]
X = normalization(X)	#||X_i||_2^2 = 1
```

- - -

### Kernels computation

MKLpy constains several functions to generate kernels for vectorial, booelan, 
and string kernels. 
The base syntax for a kernel function is `K = k(X, Z=None, **args)`, where `X` and `Z` are two matrices containing examples (rows), and `K` is the resulting kernel matrix.
Note that we use the same syntax from scikit-learn.

```python
from MKLpy.metrics.pairwise import homogeneous_polynomial_kernel
K_train = homogeneous_polynomial_kernel(Xtr, degree=2)		#k(x,z) = <x,z>**2
K_test  = homogeneous_polynomial_kernel(Xte, Xtr, degree=2)
```

In the case of MKL algorithms, we want to create a list of kernels instead of a single one. 
In the following example, we create a list of homogeneous polynomial kernels with degrees 1 up to 10, that is

```python
from MKLpy.metrics.pairwise import homogeneous_polynomial_kernel
KL_tr = [homogeneous_polynomial_kernel(Xtr, degree=d) for d in range(1,11)]
KL_te = [homogeneous_polynomial_kernel(Xte,Xtr, degree=d) for d in range(11)]
```


#### Boolean kernels

<span style="color:red">**TODO:**</span> work in progress...


#### String kernels

Several kernels for strings are defined in MKLpy.
The usage is the same of other kernel functions, with the difference that string kernels require strings as input.

```python
from MKLpy.metrics.pairwise import 	spectrum_kernel, 
									fixed_length_subsequences_kernel, 
									all_subsequences_kernel

X = ['aabb', 'abba', 'baac']
K = spectrum_kernel(X, p=2)		#computes the 2-spectrum kernel between sequences
```

Kernels for structures, as is the case of our string kernels, usually compute the dot-product between explicit embeddings `k(x,z) = <\phi(x),\phi(z)>`. 
We can directly access the explicit embeddings via specialized functions.

Differently from kernel functions, embeddings have a single sequence as input

```python
from MKLpy.metrics.pairwise import 	spectrum_embedding, 
									fixed_length_subsequences_embedding, 
									all_subsequences_embedding

s = spectrum_embedding('aaabc', p=2)	#computes the 2-spectrum embedding
```

For computational purposes, we encode string embeddings as a dictionary containing non-zero features, that is

```python
print (s)
{'aa': 2, 'ab': 1, 'bc': 1}
```


- - -

### Metrics

MKLpy has several functions to evaluate a kernel, and to compute statistrics and various metrics.
Here we show a few of examples, including

* the **margin** margin between positive and negative classes in the kernel space;
* the **radius** of the Minimum Enclosing Ball (MEB) containing data in the kernel space;
* the radius/margin **ratio**;
* the **trace** of a kernel matrix;
* the **frobenius** norm of a kernel matrix.

```python
from MKLpy import metrics
margin = metrics.margin(K,Y)	#works only in binary classification settings
radius = metrics.radius(K)
ratio  = metrics.ratio(K,Y)
trace  = metrics.trace(K)
frob   = metrics.frobenius(K)
```



!!!note
	`K` is always a squared kernel matrix, i.e. it is not the kernel computed between test and training examples.

A further important matrix is the **spectral ratio**, that defines the complexity of a kernel matrix. The high is the spectral ratio, the high is the complexity.

```python
from MKLpy import metrics
SR = metrics.spectral_ratio(K, norm=True)
```

The normalized spectral ratio has range [0,1].


- - -

### Training

Tools and utilities showed in the first part of these examples surround the core of MKLpy, i.e. the multiple kernel learning algorithms.
Different MKL algorithms are included in this framework. In the following, we show a simple procedure to train a classifier.

As already introduced in previous examples, the first step concerns the kernels computation

```python
KL = [...]	#our list of base kernels

#usually, base kernels are normalize to prevent scaling and numerical issues
from MKLpy.preprocessing import kernel_normalization
KL_norm = [kernel_normalization(K) for K in KL]

#let us divide trainig (70%) and test (30%) examples
from MKLpy.model_selection import train_test_split
KLtr,KLte,Ytr,Yte = train_test_split(KL, Y, test_size=.3, random_state=42)
```

The interface of MKL algorithms is really similar to estimators in scikit-learn.

```python
from MKLpy.algorithms import AverageMKL
#AverageMKL simply computes the average of input kernels
#It looks bad but it is a really strong baseline in MKL ;)
mkl = AverageMKL().fit(KLtr, Ytr)		#combine kernels and train a classifier
y_preds  = mkl.predict(KLte)			#predict the output class
y_scores = mkl.decision_function(KLte)	#returns the projection on the distance vector

#evaluation with scikit-learn metrics
from sklearn.metrics import accuracy_score, roc_auc_score
accuracy = accuracy_score(Yte, y_pred)
roc_auc = roc_auc_score(Yte, y_score)
print ('Accuracy score: %.4f, roc AUC score: %.4f' % (accuracy, roc_auc))
```

MKL algorithms have several hyper-parameters, that can be selected with a classical validation procedure. 
Here we show a few of examples, check the documentation for further details.
```python
from sklearn.svm import SVC
from MKLpy.algorithms import EasyMKL

#we train a soft-margin svm with the combined kernel
svm = SVC(C=100)
mkl = EasyMKL(lam=0.1, learner=svm)	#lam [0,1] is a hyper-parameter of EasyMKL
mkl = mkl.fit(KLtr, Ytr)

#perhaps we want to use a simple cross-validation...
from MKLpy.model_selection import cross_val_score
for lam in lam_values:	#[0, 0.1 ... 0.9, 1]
	mkl = EasyMKL(lam=lam, learner=svm)
	scores = cross_val_score(KLtr, Ytr, mkl, n_folds=3, scoring='accuracy')
	print (scores)	#accuracy for each fold
```

- - -

### Optimization

Some MKL algorithms, such as GRAM, rely on an iterative optimization procedure that makes the computation quite heavy.
We provide tools to control the optimization and to design search strategies by means of **callbacks** and learning rate **scheduler**.

A callback execute a pre-defined set of operations during each iteration, or when the training starts/ends.
The simplest example of callback is the *EarlyStopping*, that interrupts the optimization when a validation metric worsens.
If you want to define custom callbacks please check the documentation.

```python
from MKLpy.algorithms import GRAM
from MKLpy.scheduler  import ReduceOnWorsening
from MKLpy.callbacks  import EarlyStopping

earlystop = EarlyStopping(
	KLva, Yva, 		#validation data, KL is a validation kernels list
	patience=5,		#max number of acceptable negative steps
	cooldown=1, 	#how ofter we run a measurement, 1= every optimization step
	metric='auc',	#the metric we monitor
)

#ReduceOnWorsening automatically redure the learning rate when a worsening solution occurs
scheduler = ReduceOnWorsening()

mkl = GRAM(
	max_iter=1000, 			
	learning_rate=.01, 		
	callbacks=[earlystop],
	scheduler=ReduceOnWorsening()).fit(KLtr, Ytr)
```

!!! warning
	*EarlyStopping* may currently be computationally expensive! You may to set a high cooldown to reduce this problem.

# Basic usage

This page contains several base examples to use MKLpy and its main features.


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

A further important metric is the **spectral ratio**, that defines the empirical complexity of a kernel matrix. The high is the spectral ratio, the high is the complexity of the kernel.

```python
from MKLpy import metrics
SR = metrics.spectral_ratio(K, norm=True)
```

The normalized spectral ratio has range [0,1].


#### Alignment

The alignment measures the similarity between two kernels. 
We have several functions to compute the alignment. These functions, showed in the following example, outputs a score that represents the alignment


```python
from MKLpy import metrics

#produces the alignment between two kernels, K1 and K2
metrics.alignment(K1, K2)

#produces the alignment between the kernel K1 and the identity matrix
metrics.alignment_ID(K1)

#produces the alignment between the kernel K1 and the ideal (or optimal) kernel
metrics.alignment_yy(K1, Y)
```

where `K1` and `K2` are two sqared kernel matrices and Y is the binary labels vector

!!! note
	The ideal kernel between two examples outputs 1 if the examples belong to the same class, -1 else.
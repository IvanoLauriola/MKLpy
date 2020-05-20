# Efficient kernels generation


In the case of MKL algorithms, we may need to create list of kernels instead of a single one as the MKL algorithms require multiple kernels to perform the combination.

A list of multiple kernels can be easily computed by using list comprehensions.
In the following example, we create a list of homogeneous polynomial kernels with degrees 1 up to 10

```python
from MKLpy.metrics.pairwise import homogeneous_polynomial_kernel as hpk
KL_tr = [hpk(Xtr,     degree=d) for d in range(1,11)]
KL_te = [hpk(Xte,Xtr, degree=d) for d in range(1,11)]
```

This solution is really simple, but it leads to two big problems:

* This specific computation is not efficient at all. Each homogeneous polynomial kernel $k(x,z) = \langle x,z\rangle^d$ requires two main operations, the dot product between examples and the explonentiation.
It is easy to see that the exponentiation needs to be recomputed for each base kernels, but the result of the dot-product may be easily cached and reused for the whole list of kernels. 
* When dealing with large datasets, the available amount of memory may not be sufficient to handle the whole kernels list.

!!! note
	Note that the caching is not doable for every possible kernels list


- - -

## Generators


In order to alleviate these problems, we provide specific kernels generators providing an efficient computations and a lower memory consumption.

The same homogeneous polynomial kernels showed in the previous example may be computed with a generator:

```python
from MKLpy.generators import HPK_generator
KLtr = HPK_generator(Xtr, degrees=range(1,11), cache=True)
```

Specifically, the generator performs a lazy computation, keeping a single kernel at a time in memory. The parameter `cache` allows a faster computation with the cost of a small amount of memory.

The generators currently available in MKLpy are:

| Generator class       | Scenario and use-case | Caching |
|-----------------------|-----------------------|---------|
| Lambda_generator      | generation of different heterogeneous kernels, for instance by mixing polynomial, rbf, and other custom functions | no |
| HPK_generator         | generation of multiple homogeneous polynomial kernels with different degrees | yes |
| RBF_generator         | generation of multiple rbf kernels with different gamma values | yes |
| Multiview_generator   | generation of a specified kernel (e.g. linear) computed to each view of the input data (e.g. audio, video, or text). In this case, we have multiple feature vectors of the same example defining different views | no |

Various examples are exposed in the following snippet

```python
from MKLpy import generators
# 10 homogeneous poly. kernels 
KL_hpk = generators.HPK_generator(X, degrees=range(1,11))

# 3 rbf kernels
KL_rbf = generators.RBF_generator(X, gamma=[.001, .01, .1])

from MKLpy.metrics import pairwise
# 2 custom kernels (linear and polynomial)
ker_functions = [pairwise.linear, lambda X,Z : pairwise.polynomial(X,Z, degree=5)]
KL_mix = generators.Lambda_generator(X, ker_functions)

# 4 linear kernels from multiview
X1 = ...
X2 = ...	#each view consists of a specific group of features
X3 = ...
X4 = ...
KL_multi = Multiview_generator([X1, X2, X3, X4], kernel=pairwise.linear)
```


- - -

## Benchmarking

In the Table below we show the time and memory consumption required by EasyMKL with 20 homogeneous polynomial kernels applied to two datasets, *Madelon* and *Phishing*.


|Dataset | examples x features | list | generator w. cache | generator |
|--------|--------------------|------|--------------------|-----------|
|Madelon | 6000     x 5000     | 24.4[s], 7.3[GB]   | 28.0[s], 2.6[GB]  | 60.5[s], 2.3[GB]  |
|Phishing| 11055    x 68       | 115.7[s], 23.9[GB] | 120.4[s], 6.2[GB] | 126.8[s], 5.7[GB] |


**TL;DR** Specifically, we show the resources usage when dealing with explicit lists of kernels and MKLpy generators. 
The values include the kernels computation and the algorithm training.
What is striking from the table is the huge improvement in memory consumption (x3.2 and x4.2 reduction with *Madelon* and *Phishing* respectively), that is the main limitation of MKL algorithms. 

Additionally, the benefits of the caching mechanism, that in the case of HPKs consists of a pre-computation of a linear kernel, are evident, especially in the case of *Madelon* dataset where the computation of the dot-product is particularly demanding.


!!! see "Datasets"
	The datasets used in this experiment are freely available on <br>
	https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/

- - - 

What if I do not care about memory usage?

```python
explicit_kernels_list = my_generator.to_list()
```
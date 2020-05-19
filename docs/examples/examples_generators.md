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
However, the result of the dot-product may be easily cached and used for the whole list of kernels.
* When dealing with large datasets, the available amount of memory may not be sufficient to handle the whole kernels list.

!!! note
	Note that the caching is not doable for every possible kernels list

In order to alleviate these problems, we provide specific kernels generators.
The same homogeneous polynomial kernels may be computed as

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

What if I do not care about memory usage?

```python
explicit_kernels_list = my_generator.to_list()
```
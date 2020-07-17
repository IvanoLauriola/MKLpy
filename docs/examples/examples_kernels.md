# Kernels computation

MKLpy constains several functions to generate kernels for vectorial, booelan, 
and string kernels. 
The base syntax for a kernel function is `K = k(X, Z=None, **args)`, where `X` and `Z` are two matrices containing examples (rows), and `K` is the resulting kernel matrix.
As previously mentioned, the type of input data can be `ndarray`, `torch.Tensor`, or other iterables castable into tensors.
Note that we use the same syntax from scikit-learn.

In the following snippets, we assume that `Xtr` and `Xte` are the training and test input matrices respectively.


---

## Vectorial kernels


Several kernel functions exist to deal with vectorial data, where each example $x$ is described by a real-valued feature vector $x \in \mathbb{R}^d$.
The following table describes the kernel functions for vectorial data provided by MKLpy

| Kernel function                   | Definition              | Parameters |
|-----------------------------------|:-----------------------:| -----------|
| linear_kernel                     | $\langle x,z \rangle$ | -       |
| homogeneous_polynomial_kernel     | $\langle x,z \rangle^d$ | d: `int` |
| polynomial_kernel     | $(gamma \langle x,z \rangle + coef0)^d$ | d: `int`, gamma: `float`, coef0: `float` |
| rbf_kernel     | $exp(-gamma \|x-z\|_2^2)$ | gamma: `float` |
| euclidean_distances     | $\|x-z\|_2^2$ | - |

These kernels are available in the `MKLpy.metrics.pairwise` module. An example of invocation is shown in the following

```python
from MKLpy.metrics.pairwise import homogeneous_polynomial_kernel as hpk
K_train = hpk(Xtr, degree=2)
K_test  = hpk(Xte, Xtr, degree=2)
```


Alternatively, you can use kernel functions from the scikit-learn package
```python
from sklearn.metrics.pairwise import rbf_kernel
K_train = rbf_kernel(Xtr, gamma=.1)
```


!!! see
	Scikit-learn provides several kernel functions (that may not accept `torch.Tensor` as input). For further details see [here](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics.pairwise).







- - -

## Boolean kernels

Boolean kernels are kernel functions specifically designed for binary-valued and categorical (one-hotted) datasets.
The implicit feature space of these kernels consists of logical formulae, such as conjunctions, disjunctions, or their combinations.

Assuming $n$ be the dimension of feature vectors, boolean kernels available in MKLpy are:


| Kernel function                   | Definition              | Parameters |
|-----------------------------------|:-----------------------:| -----------|
| monotone_conjunctive_kernel       | $\binom{\langle x,z \rangle}{c}$   | c: `int` (arity of the conjunctions)  |
| monotone_disjunctive_kernel       | $\binom{n}{d}-\binom{n-\langle x,x \rangle}{d}-\binom{n-\langle z,z \rangle}{d} +\binom{n-\langle x,x \rangle-\langle z,z \rangle+\langle x,z \rangle}{d}$ | d: `int` (arity of the disjunctions)  |


These kernels work only with binary-valued examples, $x\in\{0,1\}^n$.
You may use boolean kernels with vectorial data if you apply a binarization of the features.

A simple binarized available in MKLpy is `MKLpy.preprocessing.binarization.AverageBinarizer`, that binarizes features by applying a hard-threshold based on the average values of original features.

```python
from MKLpy.preprocessing.binarization import AverageBinarizer
X = ... # my original non-binary examples matrix
binarizer = AverageBinarizer().fit(X)
X_bin = binarizer.transform(X)
```



!!! tip "Paper"
	The boolean kernels provided in MKLpy have been presented in the following paper:<br>
	Mirko Polato, Ivano Lauriola, and Fabio Aiolli: *"A novel boolean kernels family for categorical data". Entropy (2018)*

	If you use these kernels in scientific projects, please cite the aforementioned paper
	```
	@article{polato2018novel,
	  title={A novel boolean kernels family for categorical data},
	  author={Polato, Mirko and Lauriola, Ivano and Aiolli, Fabio},
	  journal={Entropy},
	  volume={20},
	  number={6},
	  pages={444},
	  year={2018},
	  publisher={Multidisciplinary Digital Publishing Institute}
	}
	```





- - -




## String kernels

Strings are structured objects consisting of ordered sequences of characters or symbols.
MKLpy provides multiple string kernelsbased on sub-structures.
In short, each feature describes the frequency (or a related measure) of the occurrence of a certain sub-structure in the input string.

The string kernels provided in the  `MKLpy.metrics.pairwise` module are summarized in the table below

| Kernel function                   | Parameters |
|-----------------------------------|-----------|
| spectrum_kernel                   | p: `int` the length of sub-structures |
| fixed_length_subsequences_kernel  | p: `int` the length of sub-structures |
| all_subsequences_kernel           | - |




The sintax is quite similar compared to other kernel functions. The only difference is that these kernels require strings as input instead of matrices.


```python
from MKLpy.metrics.pairwise import 	spectrum_kernel

X = ['aabb', 'abba', 'baac']
K = spectrum_kernel(X, p=2)
```

!!! warning
	Note that, due to the nature of strings, tensors cannot be used.


These kernels compute the explicit representation and they they perform the dot-product between the pairwise representations, i.e. $k(x,z) = \langle\phi(x),\phi(z)\rangle.

Additionally, we can directly access the explicit embeddings via specialized functions.

```python
from MKLpy.metrics.pairwise import 	spectrum_embedding, 
									fixed_length_subsequences_embedding, 
									all_subsequences_embedding

s = spectrum_embedding('aaabc', p=2)	#computes the 2-spectrum embedding
```

For computational purposes, we encode string embeddings as dictionaries containing non-zero features, i.e.

```python
print (s)
{'aa': 2, 'ab': 1, 'bc': 1}
```


!!! tip "Book"
	If you need further information concerning string kernels, you may refer to:

	Shawe-Taylor John, and Nello Cristianini. *"Kernel methods for pattern analysis". Cambridge university press* (2004).

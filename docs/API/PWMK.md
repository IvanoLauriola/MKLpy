
PWMK is a heuristic MKL algorithm that assigns the weights according to the individual kernels performance.
The algorithm runs multiple SVM (or base learners) to compute the accuracy for each base kernel.


The algorithm returns the following combination of base kernels

$$k_{\mu}(x,z)=\sum_r^P\mu_rk_r(x,z),\quad \mbox{ where } \mu_r = \frac{acc(K_r)-m\delta}{\sum_h\left(acc(K_h)-m\delta\right)}$$

Specifically, $m$ is the minimum accuracy achieved, and $\delta$ is a hyper-parameter.

!!! tip "Paper"
	If you need additional details about PWMK, please refer to the following paper:<br>
	Tanabe, H., Ho, T. B., Nguyen, C. H., & Kawasaki, S: *"Simple but effective methods for combining kernels in computational biology". IEEE International Conference on Research, Innovation and Vision for the Future in Computing and Communication Technologies (2008, July)*

```python
MKLpy.algorithms.PWMK(
	delta=.4,
	cv=3,
	**kwargs,
	)
```

=== "Parameters (__init__)"
	|**Parameter**    | Type        | Description |
	|-----------------|-------------|-------------|
	| **delta**         | *double*    | |
	| **cv**      | *int* or [cv-splitter](https://scikit-learn.org/stable/glossary.html#term-cv-splitter)   | the cross-validation used to compute the accuracy|
	| ** \*\*kwargs** | *args*      | MKL parameters, see [here](MKL.md) |

=== "Attributes"
	|**Attribute**        | Type       | Description |
	|-----------------|------------|-------------|
	| **n_kernels**   | *int*      | number of combined kernels |
	| **KL**          | *list*     | the training kernels list  |
	| **func_form**   | *callable* | the combination function (e.g. summation, average...)|
	| **solution**    | *dict*     | the solution of the optimization |



- - -

## Methods

See standard MKL methods [here](MKL.md)


- - -

## Examples

In the following example, we show two ways to use a 5-fold cv to determine the combination weights
```python
from MKLpy.algorithms import PWMK
mkl = PWMK(delta=0, cv=5).fit(KLtr, Ytr)

from sklearn.model_selection import KFold
cv = KFold(n_splits=5, shuffle=True, random_state=42)
mkl = PWMK(delta=0, cv=cv).fit(KLtr, Ytr)
```

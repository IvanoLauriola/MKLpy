
EasyMKL finds the kernels combination that maximizes the margin between classes.
A relaxation of the maximum margin problem is considered to make the problem tractable. 
The computational complexity is comparable to a single SVM.


The algorithm learns a convex combination of base kernels with form 

$$k_{\mu}(x,z)=\sum_r^P\mu_rk_r(x,z),\quad \mu_r\ge 0 \land\|\mu\|_1=1,$$

where $\mu$ is the weights vector learned by EasyMKL.

!!! tip "Paper"
	If you need additional details about EasyMKL, please refer to the following paper:<br>
	Fabio Aiolli and Michele Donini: *"EasyMKL: a scalable multiple kernel learning algorithm". Neurocomputing (2015)*

```python
MKLpy.algorithms.EasyMKL(
	lam=.1, 
	learner=MKLpy.algorithms.KOMD(lam=.1), 
	solver='auto'
	**kwargs,
	)
```

=== "Parameters (__init__)"
	|**Parameter**    | Type        | Description |
	|-----------------|-------------|-------------|
	| **lam**         | *double*    | a regularization hyper-parameter, with values in the range [0,1]. With `lam=0`, the algorithm maximizes the distance between classes, whereas with `lam=1` EasyMKL maximizes the distance between centroids.|
	| **solver**      | *str*       | solver used during the optimization, possible values are `auto`, `libsvm`, or `cvxopt` |
	| ** \*\*kwargs** | *args*      | MKL parameters, see [here](MKL.md)

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

```python
from MKLpy.algorithms import EasyMKL
mkl = EasyMKL(lam=1.)
mkl = mkl.fit(KLtr, Ytr)
```

- - -

## Citing EasyMKL

If you use EasyMKL in your scientific project, please cite the following pubblication:
```
@article{aiolli2015easymkl,
  title={EasyMKL: a scalable multiple kernel learning algorithm},
  author={Aiolli, Fabio and Donini, Michele},
  journal={Neurocomputing},
  year={2015},
  publisher={Elsevier}
}
```

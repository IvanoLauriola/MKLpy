
AverageMKL is a simple wrapper defining the combination as the average of base kernels.
Even if the average is a trivial solution, it is known to be a hard baseline in MKL. 
This wrapper helps the experimentation and the evaluation of this baseline against other complex approaches.

The kernels combination is trivially defined as

$$k_{\mu}(x,z)=\sum_r^P\mu_rk_r(x,z),\quad \mu_r=\frac{1}{P}$$

```python
MKLpy.algorithms.AverageMKL(
	learner=sklearn.svm.SVC(C=1000), 
	**kwargs,
	)
```


=== "Parameters (__init__)"
	|**Parameter**    | Type        | Description |
	|-----------------|-------------|-------------|
	| **learner**     | *Object*    | the base learner dealing with the combined kernel |
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
from MKLpy.algorithms import AverageMKL
mkl = AverageMKL()
mkl = mkl.fit(KLtr, Ytr)
```

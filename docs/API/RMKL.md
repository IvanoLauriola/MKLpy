
R-MKL (**R**adius MKL) is a `TwoStepMKL` algorithm finding the kernels combination that maximizes the margin between classes. Differently to other approaches, R-MKL regularizes the solution by introducing radius information.

The algorithm learns a convex combination of base kernels with form 

$$k_{\mu}(x,z)=\sum_r^P\mu_rk_r(x,z),\quad \mu_r\ge 0 \land\|\mu\|_1=1,$$

where $\mu$ is the weights vector learned by R-MKL.

!!! tip "Paper"
	If you need additional details about R-MKL, please refer to the following paper:<br>
	Do Huyen, Kalousis Alexandros, Woznica Adam, and Hilario Melanie: *"Margin and radius based multiple kernel learning". Joint European Conference on Machine Learning and Knowledge Discovery in Databases (2009)*


```python
MKLpy.algorithms.RMKL(
	C=1.0
	**kwargs,
	)
```


=== "Parameters (__init__)"
	|**Parameter**    | Type        | Description |
	|-----------------|-------------|-------------|
	| **C**           | double      | The SVM cost for non-separable data |
	| ** \*\*kwargs** | *args*      | TwoStepMKL parameters, see [here](MKL.md)

=== "Attributes"
	|**Attribute**        | Type       | Description |
	|-----------------|------------|-------------|
	| **n_kernels**   | *int*      | number of combined kernels |
	| **KL**          | *list*     | the training kernels list  |
	| **func_form**   | *callable* | the combination function (e.g. summation, average...)|
	| **solution**    | *dict*     | the solution of the optimization |
	| **direction**   | *str*      | direction of the optimization, `min` or `max` |
	| **convergence** | *bool*     | `True` iff the algorithm reaches the convergence |
	| **cache**       | *dict*     | a dictionary containing intermediate results and data structures used to speed-up the computation |


- - - 

## Methods

See standard TwoStepMKL methods [here](MKL.md)


- - - 

## Examples

```python
from MKLpy.algorithms import RMKL
from MKLpy.scheduler  import ReduceOnWorsening
from MKLpy.callbacks  import EarlyStopping

earlystop = EarlyStopping(
	KLva, Yva, 		#validation data, KL is a validation kernels list
	patience=5,		#max number of acceptable negative steps
	cooldown=1, 	#how ofter we run a measurement, 1 means every optimization step
	metric='roc_auc',	#the metric we monitor
)

#ReduceOnWorsening automatically redure the learning rate when a worsening solution occurs
scheduler = ReduceOnWorsening()

mkl = RMKL(
	max_iter=1000, 			
	learning_rate=.1, 		
	callbacks=[earlystop],
	scheduler=ReduceOnWorsening()).fit(KLtr, Ytr)
```

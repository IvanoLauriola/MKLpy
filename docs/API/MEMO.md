
MEMO (**M**inimum **E**ffort **M**aximum output) is a `TwoStepMKL` algorithm finding the kernels combination that simultaneously maximizes the margin between classes while minimizes the empirical complexity of the resulting kernel.

The algorithm learns a convex combination of base kernels with form 

$$k_{\mu}(x,z)=\sum_r^P\mu_rk_r(x,z),\quad \mu_r\ge 0 \land\|\mu\|_1=1,$$

where $\mu$ is the weights vector learned by MEMO.

!!! tip "Paper"
	If you need additional details about MEMO, please refer to the following paper:<br>
	Ivano Lauriola, Mirko Polato, and Fabio Aiolli: *"The minimum effort maximum output principle applied to Multiple Kernel Learning". ESANN (2018)*

```python
MKLpy.algorithms.MEMO(
	theta=0.0,
	min_margin=1e-4,
	solver='auto',
	**kwargs,
	)
```


=== "Parameters (__init__)"
	|**Parameter**    | Type        | Description |
	|-----------------|-------------|-------------|
	| **theta**       | double      | This hyper-parameter regularizes between margin maximization ($\theta=0$) and complexity minimization ($\theta=\infty$)|
	| **min_margin**  | double      | the minimum admissible margin |
	| **solver**      | *str*       | solver used during the optimization, possible values are `auto`, `libsvm`, or `cvxopt` |
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
from MKLpy.algorithms import MEMO
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

mkl = MEMO(
	max_iter=1000, 			
	learning_rate=.1, 		
	callbacks=[earlystop],
	scheduler=ReduceOnWorsening()).fit(KLtr, Ytr)
```

- - -

## Citing MEMO

If you use MEMO in your scientific project, please cite the following pubblication:
```
@inproceedings{lauriola2018minimum,
  title={The minimum effort maximum output principle applied to Multiple Kernel Learning.},
  author={Lauriola, Ivano and Polato, Mirko and Aiolli, Fabio},
  booktitle={ESANN},
  year={2018}
}

```


GRAM (**G**radient-based **RA**dius-**M**argin optimization) is a `TwoStepMKL` algorithm finding the kernels combination that simultaneously maximizes the margin between classes while minimizes the radius of the resulting kernel.

The algorithm learns a convex combination of base kernels with form 

$$k_{\mu}(x,z)=\sum_r^P\mu_rk_r(x,z),\quad \mu_r\ge 0 \land\|\mu\|_1=1,$$

where $\mu$ is the weights vector learned by GRAM.

!!! tip "Paper"
	If you need additional details about GRAM, please refer to the following paper:<br>
	Ivano Lauriola, Mirko Polato, and Fabio Aiolli: *"Radius-Margin Ratio Optimization for Dot-Product Boolean Kernel Learning". ICANN (2017)*

```python
MKLpy.algorithms.GRAM(
	**kwargs,
	)
```


=== "Parameters (__init__)"
	|**Parameter**    | Type        | Description |
	|-----------------|-------------|-------------|
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
from MKLpy.algorithms import GRAM
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

mkl = GRAM(
	max_iter=1000, 			
	learning_rate=.01, 		
	callbacks=[earlystop],
	scheduler=ReduceOnWorsening()).fit(KLtr, Ytr)
```

!!! warning
	If the learning rate is too high or if the initial solution (i.e. the simple average) is not separable, you may have numerical errors.


- - -

## Citing GRAM

If you use GRAM in your scientific project, please cite the following pubblication:
```
@inproceedings{lauriola2017radius,
  title={Radius-margin ratio optimization for dot-product boolean kernel learning},
  author={Lauriola, Ivano and Polato, Mirko and Aiolli, Fabio},
  booktitle={International conference on artificial neural networks},
  pages={183--191},
  year={2017},
  organization={Springer}
}
```

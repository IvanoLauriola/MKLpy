#MKL

This is an *abstract* class describing the main structure of MKL algorithms.
Input checks and the general behavior of MKL algorithms are implemented here.
The combining mechanism and te optimization is implemented in the derived sub-classes

```python
MKLpy.algorithms.MKL(
	multiclass_strategy='ova', 
	verbose   = False, 
	tolerance = 1e-7, 
	learner   = None,
	max_iter  = -1,
)
```

!!! warning
	```MKLpy.algorithms.MKL``` is only an abstract class providing a general structure to other MKL algorithms. Do not use this class for doing classification!


=== "Parameters (__init__)"
	|**Parameter**    | Type        | Description |
	|-----------------|-------------|-------------|
	| **multiclass_strategy**| *str*| the meta algorithm for solving multiclass problems (`ova` or `ovr`) |
	| **verbose**     | *boolean*   | verbosity level  |
	| **tolerance**   | *double*    | the numerical tolerance during optimization |
	| **learner**     | *Object*    | the base learner dealing with the combined kernel |
	| **max_iter**    | *int*       | the maximum number of iterations, -1 means no limit |

=== "Attributes"
	|**Attribute**        | Type       | Description |
	|-----------------|------------|-------------|
	| **n_kernels**   | *int*      | number of combined kernels |
	| **KL**          | *list*     | the training kernels list  |
	| **func_form**   | *callable* | the combination function (e.g. summation, average...)|
	| **solution**    | *dict*     | the solution of the optimization |


- - -

## Methods

**.fit**: finds the kernels combination and trains the base learner when available

```python
fit(KL, Y)
```

* **KL**: a list of training kernel matrices with the same shape
* **Y**: the labels

Returns: `self`

- - -

**.combine_kernels**: combine the input kernels

```python
combine_kernels(KL, Y)
```

* **KL**: a list of training kernel matrices with the same shape
* **Y**: the labels

Returns: the [solution](404.md) of the algorithm

- - -

**.predict**: predicts the labels given a test kernels list, i.e. kernels computed between test and training examples

```python
preidct(KLte)
```

* **KLte**: a list of test kernel matrices. The dimension of kernels has to match with the training size, i.e. ```cols(KLte[i]) == rows(KL[i]) == cols(KLtr[i])```

Returns: a mono-dimensional array containing test predictions

- - -

**.decision_function**: projects the test examples on the class-distance vector. The high is the value the low is the distance with the positive class.

```python
decision_function(KLte)
```

* **KLte**: a list of test kernel matrices. The dimension of kernels has to match with the training size, i.e. ```cols(KLte[i]) == rows(KL[i]) == cols(KLtr[i])```

Returns: a mono-dimensional array containing test scores

!!! warning
	*.decision_function* may not work in multiclass scenarios


- - -

#TwoStepMKL(MKL)

Two-step algorithms rely on an alternate optimization procedure, where they alternatively optimize, step by step, the combination weights and the SVM parameters until convergence.

MKLpy provides an *abstract* class, namely `TwoStepMKL`, that extends `MKL` and encapsulates the alternate optimization strategy.

!!! warning
	```MKLpy.algorithms.TwoStepMKL``` is only an abstract class providing a general structure to other TwoStepMKL algorithms. Do not use this class for doing classification!

=== "Parameters (__init__)"
	|**Parameter**    | Type        | Description |
	|-----------------|-------------|-------------|
	| **learning_rate** | *double*  | the step used in gradient descent optimization |
	| **callbacks** | *list* | a list of [callbacks](404.md) used every iteration |
	| **scheduler** | *Scheduler* | the [scheduling strategy](404.md) used to play with the learning rate |
	| **max_iter**    | *int*       | the maximum number of iterations, -1 means no limit |
	| ** \*\*kwargs** | *args*      | MKL parameters, see [here](MKL.md)|


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

The method `combine_kernels` is overrided from `MKL` to implement the dual optimization meta-algorithm.

Moreover, `TwoStepMKL` has a few methods in addition to the ones derived from `MKL`.

!!! warning
	These are abstract methods needed to implement novel solutions, and they are used in the optimization process. If you just need to use existing MKL algorithms, do not consider these methods.


**.initialize_optimization**: this method initialize the optimization context, including algorithm-specific data structures, and it returns the initial [solution](404.md)

```python
initialize_optimization()
```

Returns: `self`

- - -

**.do_step**: his method computes an optimization step abd returns the new (updated) solution.

```python
do_step(sol)
```

* **sol**: the current [solution](404.md)

- - -


**.score**: projects the test examples on the class-distance vector. The high is the value the low is the distance with the positive class.

This method is designed to leverage the internal SVM used by `TwoStepMKL` algorithms during the optimization.

```python
score(KLte)
```

* **KLte**: a list of test kernel matrices. The dimension of kernels has to match with the training size, i.e. ```cols(KLte[i]) == rows(KL[i]) == cols(KLtr[i])```

Returns: a mono-dimensional array containing test scores


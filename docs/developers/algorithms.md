
## <span style="color:red">**MKL**</span>

This is an *abstract* class describing the main structure of MKL algorithms.
Input checks and the general behavior of MKL algorithms are implemented here.
The combining mechanism and te optimization is implemented in the derived sub-classes

```python
MKLpy.algorithms.MKL(learner, multiclass_strategy)
```

* **learner**: a kernel machine scikit compliant
* **multiclass_strategy**: `ovo` for one-vs-one, `ova` for one-vs-all

#### Attributes

* **n_kernels** (*int*): number of combined kernels
* **KL** (*iterable*): kernels list
* **func_form** (*callable*): the combination function (e.g. summation)
* **solution** (*Solution*): the solution

- - -

#### Methods

**.fit**: finds the kernels combination and trains the base learner 

```python
fit(KL, Y)
```

* **KL**: a list of training kernel matrices with the same shape
* **Y**: the labels

Returns: *self*

- - -

**.combine_kernels**: combine the input kernels

```python
combine_kernels(KL, Y)
```

* **KL**: a list of training kernel matrices with the same shape
* **Y**: the labels

Returns: the combined kernel matrix

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

Returns: a mono-dimensional array containing test predictions

!!! warning
	*.decision_function* may not work in multiclass scenarios

- - -

## <span style="color:red">**AverageMKL**</span> (MKL)

AverageMKL is a simple wrapper that defines the combination as the average of base kernels.
Even if the average is a trivial solution, it is known to be a hard baseline in MKL. 
This wrapper helps the experimentation and the evaluation of this baseline against other complex approaches.

The kernels combination is defined as
<img src="https://latex.codecogs.com/svg.latex?\Large&space;k_{\mu}(x,z)=\sum_r^P\mu_rk_r(x,z),\quad \mu_r=\frac{1}{P}" /img>

```python
MKLpy.algorithms.AverageMKL(learner=sklearn.svm.SVC(C=1000), multiclass_strategy='ova')
```

* **learner**: a kernel machine scikit compliant.
* **multiclass_strategy**: `ovo` for one-vs-one, `ova` for one-vs-all.


#### Examples

```python
from MKLpy.algorithms import AverageMKL
mkl = AverageMKL()
mkl = mkl.fit(KLtr, Ytr)
```


- - -

## <span style="color:red">**EasyMKL**</span> (MKL)

EasyMKL find the kernels combination that maximizes the margin between classes.
A relaxation of the maximum margin problem is considered to make the problem tractable. 
The computational complexity is comparable to a single SVM.


Also in this case, the algorithm learns a convex combination of base kernels with form 
<img src="https://latex.codecogs.com/svg.latex?\Large&space;k_{\mu}(x,z)=\sum_r^P\mu_rk_r(x,z),\ \mu_r\ge 0 \land\|\mu\|_1=1" /img>, 
where <img src="https://latex.codecogs.com/svg.latex?\Large&space;\mu" /img> is the weights vector learned by EasyMKL.

```python
MKLpy.algorithms.EasyMKL(
	lam=.1, 
	learner=MKLpy.algorithms.KOMD(lam=.1), 
	multiclass_strategy='ova')
```

* **lam**: a regularization hyper-parameter, with values in the range [0,1].
With `lam=0`, the algorithm maximizes the distance between classes, whereas with `lam=1` EasyMKL maximizes the distance between centroids.
* **learner**: a kernel machine scikit compliant. According to the EasyMKL paper. we use `KOMD` as default choice.
* **multiclass_strategy**: `ovo` for one-vs-one, `ova` for one-vs-all


#### Examples

```python
from MKLpy.algorithms import EasyMKL
mkl = EasyMKL(lam=1.)
mkl = mkl.fit(KLtr, Ytr)
```

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

- - -

## <span style="color:red">**GRAM**</span> (MKL)

GRAM (**G**radient-based **RA**dius-**M**argin optimization) find the kernels combination that simultaneously maximizes the margin between classes while minimizes the radius of the resulting kernel.

This algorithm needs to solve multiple QP problems, and you may need to play with learning rates and other parameters to make the training tractable.


The algorithm learns a convex combination of base kernels with form 
<img src="https://latex.codecogs.com/svg.latex?\Large&space;k_{\mu}(x,z)=\sum_r^P\mu_rk_r(x,z),\ \mu_r\ge 0 \land\|\mu\|_1=1" /img>, 
where <img src="https://latex.codecogs.com/svg.latex?\Large&space;\mu" /img> is the learned weights vector.

```python
MKLpy.algorithms.GRAM(
	learner=sklearn.svm.SVC(C=1000), 
	multiclass_strategy='ova',
	max_iter=1000, 
	learning_rate=0.01, 
	callbacks=[], 
	scheduler=None
	)
```

* **learner**: a kernel machine scikit compliant.
* **multiclass_strategy**: `ovo` for one-vs-one, `ova` for one-vs-all.
* **max_iter**: maximum number of iterations.
* **learning_rate**: the learning rate of the optimization procedure. 
* **callbacks**: a list of `Callback` objects.
* **scheduler**: the scheduler for the learning rate.

!!! see also
	See also the sections `Callback` and `Scheduler`

#### Examples

```python
from MKLpy.algorithms import GRAM
from MKLpy.scheduler  import ReduceOnWorsening
from MKLpy.callbacks  import EarlyStopping

earlystop = EarlyStopping(
	KLva, Yva, 		#validation data, KL is a validation kernels list
	patience=5,		#max number of acceptable negative steps
	cooldown=1, 	#how ofter we run a measurement, 1= every optimization step
	metric='auc',	#the metric we monitor
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

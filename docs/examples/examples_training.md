# Training


MKLpy provides multiple MKL algorithms that are grouped into two categories, i.e.

* **OneStepMKL**: these algorithms learn the combination weights and the base learner (e.g. a SVM using the combined kernel) in a single pass. Usually, the kernels combination and the base learner training phases are divided.
* **TwoStepMKL**: these algorithms use an iterative optimization procedure where each iteration updates the combination weights while fixing the base learner parameters, and then updates the base learner parameters while fixing the parameters of the combination. This procedure is repeated until convergence.

These two different strategies are individually implemented in MKLpy. A few examples are shown in the remainder of this tutorial.


- - -

## A primer example

Tools and utilities showed in the previous tutorials surround the core of MKLpy, i.e. the multiple kernel learning algorithms.
In the following, we show a simple procedure to train a classifier from the `MKLpy.algorithms` sub-package.

As already introduced in previous examples, the first step concerns the kernels computation



```python
KL = [...]	#our list of base kernels (or a generator)

#usually, base kernels are normalize to prevent scaling and numerical issues
from MKLpy.preprocessing import kernel_normalization
KL_norm = [kernel_normalization(K) for K in KL]

#let us divide trainig (70%) and test (30%) examples
from MKLpy.model_selection import train_test_split
KLtr, KLte, Ytr, Yte = train_test_split(KL, Y, test_size=.3, random_state=42)
```

The interface of MKL algorithms is really similar to estimators in scikit-learn.
The highlighted line corresponds to the instantiation of a MKL model and the training.

```python hl_lines="4"
from MKLpy.algorithms import AverageMKL
#AverageMKL simply computes the average of input kernels
#It looks bad but it is a really strong baseline in MKL ;)
mkl = AverageMKL().fit(KLtr, Ytr)		#combine kernels and train the classifier
y_preds  = mkl.predict(KLte)			#predict the output class
y_scores = mkl.decision_function(KLte)	#returns the projection on the distance vector
```

We can evaluate a solution leveraging scikit-learn tools

```python
#evaluation with scikit-learn metrics
from sklearn.metrics import accuracy_score, roc_auc_score
accuracy = accuracy_score(Yte, y_pred)
roc_auc = roc_auc_score(Yte, y_score)
print ('Accuracy score: %.4f, roc AUC score: %.4f' % (accuracy, roc_auc))
```

!!! see
	Scikit-learn provides several evaluation metrics. For further details see [here](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics).








- - -

## OneStepMKL

OneStepMKL algorithms are usually faster than other solutions as they need to execute a single optimization step (that may require the solution of a single QP optimization problem).

Currently, the OneStepMKL algorithms here available are

|Name       | Hyper-parameters | Default base learner | Source |
|-----------|------------|----------------------|:------:|
| AverageMKL| -          | `sklearn.svm.SVC(C=1000)`          |  -     |
| EasyMKL   | $\lambda\in [0,1]$  | `MKLpy.algorithms.KOMD(lam=0.1)` | [[1]](https://www.sciencedirect.com/science/article/abs/pii/S0925231215003653) |


The main characteristic of OneStepMKL algorithm is that they deal only with the kernels combination. 
After that step, a classical kernel method (that we call base learner) is required to solve the task with the combined kernel.
A default base learner is already set, but you can specify a different one when initializing the algorithm.


```python
from sklearn.svm import SVC
base_learner = SVC(C=100)

from MKLpy.algorithms import EasyMKL
mkl = EasyMKL(lam=0.1, learner=base_learner)
mkl = mkl.fit(KLtr, Ytr)
```

The same applies for `AverageMKL`
```python
from MKLpy.algorithms import AverageMKL
mkl = AverageMKL(learner=base_learner).fit(KLtr, Ytr)
```


Alternatively, we can use the MKL just to learn the kernels combination, without training the base learner

```python
mkl = EasyMKL()
ker_matrix = mkl.combine_kernels(KLtr, Ytr)
```


- - -

## TwoStepMKL

Some MKL algorithms, such as GRAM, rely on an iterative optimization procedure that makes the computation quite heavy.

Currently, the TwoStepMKL algorithms here available are

|Name      | Hyper-parameters | Source |
|----------|------------|:------:|
| GRAM     | -          | [[2]](https://www.researchgate.net/publication/318468451_Radius-Margin_Ratio_Optimization_for_Dot-Product_Boolean_Kernel_Learning)     |
| R-MKL    | $C \ge 0$        |[[3]](https://link.springer.com/content/pdf/10.1007/978-3-642-04180-8_39.pdf)  |
| MEMO     | $\theta \ge 0$   |[[4]](https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es2018-181.pdf) |

Differently from the previous methods, TwoStepMKL algorithms do not need a base learner dealing with the combination. 
Indeed, these algorithms are designed together with a SVM-based classifier, and they learn the hyper-plane used for doing classification with the kernels combination weights.
However, you can always specify a different base learner according to your needs ;).

```python
from MKLpy.algorithms import GRAM, MEMO, RMKL
from sklearn.svm import SVC
mkl = MEMO(theta=10)
mkl = MEMO(theta=10, learner=SVC(C=100))
```

We provide tools to control the optimization and to design search strategies by means of **callbacks** and learning rate **scheduler**.
A callback execute a pre-defined set of operations during each iteration, or when the training starts/ends.
Currently, we provide two callbacks, that are

* *EarlyStopping*, that interrupts the optimization when a validation metric worsens;
* *Monitor*, that monitors some statistics during optimization, such as the objective function and the combination weights.

If you want to define custom callbacks please check the documentation.

```python
from MKLpy.algorithms import GRAM
from MKLpy.scheduler  import ReduceOnWorsening
from MKLpy.callbacks  import EarlyStopping, Monitor

monitor   = Monitor()
earlystop = EarlyStopping(
	KLva, Yva, 		#validation data, KL is a validation kernels list
	patience=5,		#max number of acceptable negative steps
	cooldown=1, 	#how ofter we run a measurement, 1 means every optimization step
	metric='rocauc',#the metric we monitor, roc_auc or accuracy
)

#ReduceOnWorsening automatically reduces the 
#learning rate when a worsening solution occurs
scheduler = ReduceOnWorsening()

mkl = GRAM(
	max_iter=1000, 			
	learning_rate=.01, 		
	callbacks=[earlystop, monitor],
	scheduler=scheduler).fit(KLtr, Ytr)
```



!!! warning
	If you use *EarlyStopping* with a custom base learner, the training may be computationally expensive! 
	In that case, the callback needs to train the base learner each iteration in order to do predictions (and thus evaluation). 
	You may to set a high cooldown to reduce this problem. Alternatively, you can just use the default base learner.


!!! note
	Note that the method `.combine_kernels(...)` is not available in TwoStepMKL algorithms.

- - -

## Multiclass classification

The MKL algorithms in this library support the multiclass classification by using two decomposition strategies, that are

* one-vs-all (or one-vs-rest): the main task is decomposed in several binary classification sub-tasks. Each sub-task uses examples from one class as positive training data, and all of the other examples as negatives. 

* one-vs-one: the main task is decomposed in several binary classification sub-tasks. Each sub-task uses only examples from two classes.

A voting scheme is then applied to elect the resulting class.

These strategies are automatically used when multiclass data is recognized as input of a MKL algorithm.<br>
We just need to specify the multiclass strategy, `ovo` for one-vs-one, `ova` (default) for one-vs-all.

```python
from MKLpy.algorithms import AverageMKL
mkl = AverageMKL(multiclass_strategy='ova').fit(KLtr, Ytr)
```

Alternatively, we can use a *MetaClassifier*, that is a multiclass wrapper.
```python
from MKLpy.algorithms import EasyMKL
from MKLpy.multiclass import OneVsRestMKLClassifier, OneVsOneMKLClassifier
mkl = EasyMKL(lam=.1)
clf = multiclass.OneVsRestMKLClassifier(mkl).fit(KLtr, Ytr)
```

If you want to play with different multiclass strategies or if you want to analyze different voting rules, you can just work on your custom implementation by redefining some abstract methods of `OneVs[Rest/One]MKLClassifier`.



!!! tip "See also..."
	If you need further details concerning MKL and decomposition multiclass strategies, see the paper:<br>
	*Ivano Lauriola et al., "Learning Dot Product Polynomialsfor multiclass problems". ESANN 2017.*


# Training


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
The highlighted line corresponds to the instantiation of a MKL algorithm and the training.

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




MKL algorithms have several hyper-parameters, that can be selected with a classical validation procedure. 
Here we show a few of examples, check the documentation for further details.
```python
from MKLpy.algorithms import EasyMKL
mkl = EasyMKL(lam=0.1, learner=svm)	#lam [0,1] is a hyper-parameter of EasyMKL
mkl = mkl.fit(KLtr, Ytr)

#perhaps we want to use a simple cross-validation...
from MKLpy.model_selection import cross_val_score
for lam in lam_values:	#[0, 0.1 ... 0.9, 1]
	mkl = EasyMKL(lam=lam, learner=svm)
	scores = cross_val_score(KLtr, Ytr, mkl, n_folds=3, scoring='accuracy')
	print (scores)	#accuracy for each fold
```


- - -

## Customizations

Sometimes we may need to customize the MKL pipeline, adapting it to our task and our needs.

The first cumization concerns the selection of a different kernel machine working with the combined kernel

```python
from sklearn.svm import SVC
svm = SVC(C=100)
mkl = EasyMKL(lam=0.1, learner=svm).fit(KLtr, Ytr)
```

Alternatively, we can use the MKL just to learn the kernels combination, without training a kernel machine

```python
ker_matrix = EasyMKL().combine_kernels(KLtr, Ytr)
```


- - -

## Optimization

Some MKL algorithms, such as GRAM, rely on an iterative optimization procedure that makes the computation quite heavy.
We provide tools to control the optimization and to design search strategies by means of **callbacks** and learning rate **scheduler**.

A callback execute a pre-defined set of operations during each iteration, or when the training starts/ends.
The simplest example of callback is the *EarlyStopping*, that interrupts the optimization when a validation metric worsens.
If you want to define custom callbacks please check the documentation.

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

#ReduceOnWorsening automatically reduces the 
#learning rate when a worsening solution occurs
scheduler = ReduceOnWorsening()

mkl = GRAM(
	max_iter=1000, 			
	learning_rate=.01, 		
	callbacks=[earlystop],
	scheduler=ReduceOnWorsening()).fit(KLtr, Ytr)
```

!!! warning
	*EarlyStopping* may currently be computationally expensive! You may to set a high cooldown to reduce this problem.


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



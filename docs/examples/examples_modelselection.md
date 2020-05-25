# Model selection


MKL algorithms may have different hyper-parameters that need to be tuned. 
This section introduces a few tools to select hyper-parameters and to simplify the validation process.



## Train/test split

If we just want to divide our kernels list `KL` (and our labels vector `Y`) into training and test lists we can use the `MKLpy.model_selection.train_test_split` method

```python
from MKLpy.model_selection import train_test_split
KL = [...]	#kernels list
Y = ...		#labels 
KLtr, KLte, Ytr, Yte = train_test_split(KL, Y, random_state=42, shuffle=True, test_size=.3)
```

!!! warning
	`MKLpy.model_selection.train_test_split` provides a simple way to split a kernels list into a training and validation/test lists. 
	However, this approach is not efficient.
	If you need to reduce the time required for kernels computation, consider to directly build a training and a validation/test lists instead of using this method.






## Cross validation

However, the previous approach is really limited, and it can be used only for simple experimentations.

If we need something more complex, MKLpy provides simple routines to perform a cross-validation.
In the following example, we perform a 3-fold cross-validation with EasyMKL (with default hyper-parameters)

```python
from MKLpy.model_selection import cross_val_score
from MKLpy.algorithms import EasyMKL
mkl = EasyMKL()
scores = cross_val_score(KL, Y, mkl, n_folds=3, scoring='accuracy')
print (scores)	#accuracy for each fold
```


Finally, we can leverage the cross-validation to find the best hyper-parameters configuration.
In the following example, we use a grid-search to select the best $\lambda$ for EasyMKL and $C$ for the base SVM.


```python
from MKLpy.model_selection import cross_val_score
from MKLpy.algorithms import EasyMKL
from sklearn.svm import SVC
from itertools import product

KL, Y = ..., ...

lam_values = [0, 0.1, 0.2, 1]
C_values   = [0.01, 1, 100]
for lam, C in product(lam_values, C_values):
	svm = SVC(C=C)
	mkl = EasyMKL(lam=lam, learner=svm)
	scores = cross_val_score(KL, Y, mkl, n_folds=3, scoring='roc_auc')
	print (lam, C, scores)
```

The scoring mechanisms currently available are `accuracy`, `roc_auc`, and `f_score`.


- - -

## Playing with folds

If you need to run a simple cross-validation, you can just specify an integer value with the `n_folds` parameter. 
Otherwise, if you need more control on the validation process, you can pass a splitter


```python
from sklearn.model_selection import StratifiedKFold, LeaveOneOut

loo = LeaveOneOut()
mkl = AverageMKL()
scores = cross_val_score(KL, Y, mkl, n_folds=loo, scoring='accuracy')
```

!!! see
	Scikit-learn provides several splitters [here](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection). 
	Choose the most appropriate. 


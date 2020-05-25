# Get started!


The backend of MKLpy and its pipeline is designed to work with PyTorch tensors, leveraging their usability and efficiency in several matrix operations. 

At the same time, MKLpy takes inspiration by the scikit-learn project, and it is compliant with most of scikit functionalities, including data reading, evaluation tools, and some preprocessing transformations.
In order to maintain these features, the public interface of MKLpy accepts as input both `numpy.ndarray`, `torch.Tensor`, and other iterables castable into torch tensors. However, the backend is fully developed with a mixed combination of PyTorch and CVXOPT.

In this first tutorial, we'll show preliminaries and methods to read and preprocess input data.

- - -


## Loading data


In the simplest case, datasets can be loaded by means of popular scikit-learn routines, that are able to read files in the `svmlight` format, returning a `ndarray`.


```python 
from sklearn.datasets import load_breast_cancer
ds = load_breast_cancer()
X,Y = ds.data, ds.target

from sklearn.datasets import load_svmlight_file
file_path = ... # <- insert the file path you want to read
X,Y = load_svmlight_file(file_path)
X = X.toarray()	#be sure your ndarray is not sparse!
```

Otherwise, you can use different reading tools, such as `torch.load`

```python 
import torch
X, y = torch.load(file_path)
```

List-ceptions `[[row_1],...,[row_n]]` are also accepted. However, everything will became a tensor in the MKL pipeline.


!!! note
	Things get more complicated when dealing with structured data. We'll show some examples in the next tutorials.


- - -

## Preprocessing

MKLpy provides several tools to pre-process input data, such as


```python
from MKLpy.preprocessing import \
	normalization, rescale_01, rescale, centering
X = rescale_01(X)		#feature scaling in [0,1]
X = normalization(X)	#row (example) normalization ||X_i||_2^2 = 1
```

!!! warning
	The  transformations automatically cast the input into tensors.

Alternatively, if you deal with `ndarray`, you can use scikit functionalities

```python
from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(X)
```



!!! see
	Scikit-learn provides a plethora of pre-processing tools. You can find additional details [here](https://scikit-learn.org/stable/modules/preprocessing.html)

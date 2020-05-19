# Evaluating a kernel




The module `MKLpy.metrics` provides several functions to evaluate a kernel, and to compute statistrics and various metrics.
Here we show a few of examples, including

* the **margin** margin between positive and negative classes in the kernel space;
* the **radius** of the Minimum Enclosing Ball (MEB) containing data in the kernel space;
* the radius/margin **ratio**;
* the **trace** of a kernel matrix;
* the **frobenius** norm of a kernel matrix.

```python
from MKLpy import metrics
#we assume K be the kernel matrix and Y be the labels vector
margin = metrics.margin(K,Y)	#works only in binary classification settings
radius = metrics.radius(K)
ratio  = metrics.ratio(K,Y)
trace  = metrics.trace(K)
frob   = metrics.frobenius(K)
```



!!!note
	`K` is always a squared kernel matrix, i.e. it is not the kernel computed between test and training examples.


- - - 


## Spectral Ratio


An additional important metric is the **Spectral Ratio**, that reflects the empirical complexity of a kernel matrix. 
The Spectral Ratio is defined as the ratio between the trace of a kernel and its Frobenius norm, i.e. $\mathcal{C}(K) = \frac{\sum_i K_{i,i}}{\sqrt{\sum_{i,j} K_{i,j}^2}} = \frac{\|K\|_T}{\|K\|_F}$.

The high is the spectral ratio, the high is the complexity of the kernel.

```python
from MKLpy import metrics
SR = metrics.spectral_ratio(K, norm=True)
```

The normalized spectral ratio has range [0,1].

!!! tip "Paper"
	An exhaustive description of the Spectral Ratio is available in the following paper:<br>
	Michele Donini and Fabio Aiolli: *"Learning deep kernels in the space of dot product polynomials". Machine Learning (2017)*



- - -

## Alignment

The alignment measures the similarity between two kernels. 
We have several functions to compute the alignment. These functions, showed in the following example, outputs a score that represents the alignment


```python
from MKLpy import metrics

#produces the alignment between two kernels, K1 and K2
metrics.alignment(K1, K2)

#produces the alignment between the kernel K1 and the identity matrix
metrics.alignment_ID(K1)

#produces the alignment between the kernel K1 and the ideal (or optimal) kernel
metrics.alignment_yy(K1, Y)
```

where `K1` and `K2` are two sqared kernel matrices and Y is the binary labels vector

!!! note
	The ideal kernel between two examples outputs 1 if the examples belong to the same class, -1 else.
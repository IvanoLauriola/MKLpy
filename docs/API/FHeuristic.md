
FHeuristic is a heuristic MKL algorithm that assigns the weights according to the individual alignment with the ideal kernel.
The algorithm is particularly efficient as it does not need to solve any optimization problem with the exception of the final SVM training.


The algorithm returns the following combination of base kernels

$$k_{\mu}(x,z)=\sum_r^P\mu_rk_r(x,z),\quad \mbox{ where } \mu_r = \frac{A(\textbf{K}_r, \textbf{yy}^\top)}{\sum_hA(\textbf{K}_h, \textbf{yy}^\top)}$$

In the aforementioned definition, $A(\textbf{K1}, \textbf{K2})$ represents the alignment between the two kernel matrices, and $\textbf{yy}^\top$ is the ideal kernel matrix ($\textbf{yy}^\top_{i,j}= 1$ iff the $i$-th and $j$-th examples belong to the same class, -1 else).


!!! tip "Paper"
	If you need additional details about FHeuristic, please refer to the following paper:<br>
	S. Qiu and T. Lane: *"A Framework for Multiple Kernel Support Vector Regression and Its Applications to siRNA Efficacy Prediction". IEEE/ACM Transactions on Computational Biology and Bioinformatics (2009)*

```python
MKLpy.algorithms.FHeuristic(
	**kwargs,
	)
```

=== "Parameters (__init__)"
	|**Parameter**    | Type        | Description |
	|-----------------|-------------|-------------|
	| ** \*\*kwargs** | *args*      | MKL parameters, see [here](MKL.md) |

=== "Attributes"
	|**Attribute**        | Type       | Description |
	|-----------------|------------|-------------|
	| **n_kernels**   | *int*      | number of combined kernels |
	| **KL**          | *list*     | the training kernels list  |
	| **func_form**   | *callable* | the combination function (e.g. summation, average...)|
	| **solution**    | *dict*     | the solution of the optimization |



- - -

## Methods

See standard MKL methods [here](MKL.md)


- - -

## Examples

```python
from MKLpy.algorithms import FHeuristic
mkl = FHeuristic().fit(KLtr, Ytr)
```

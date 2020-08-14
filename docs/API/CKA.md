
CKA is a MKL algorithm that optimizes the centered alignment betwen the combined kernel matrix $\textbf{K}_\mu$ and the ideal kernel $\textbf{yy}^\top$ ($\textbf{yy}^\top_{i,j}= 1$ iff the $i$-th and $j$-th examples belong to the same class, -1 else).
The solution of the algorithm is efficiently computed in closed form.

The algorithm returns the following combination of base kernels

$$k_{\mu}(x_r,x_s)=\sum_r^P\mu_rk_r(x_r,x_s),\quad \mu = \frac{\textbf{M}^{-1}\textbf{a}}{\|\textbf{M}^{-1}\textbf{a}\|_2},$$

where $\textbf{M}_{ij} = \langle \textbf{K}_i,\textbf{K}_j \rangle_F$ and $\textbf{a}_i = \langle \textbf{K}_i,\textbf{yy}^\top \rangle_F$.
$\textbf{K}^c$ denotes the centered kernel, whereas $\langle \textbf{K}_i,\textbf{K}_j \rangle_F = \sum_r\sum_s k_i(x_r,x_s)k_j(x_r,x_s)$


!!! tip "Paper"
	If you need additional details about FHeuristic, please refer to the following paper:<br>
	Cortes, C., Mohri, M., & Rostamizadeh, A.. *"Two-stage learning kernel algorithms." (2010)*

```python
MKLpy.algorithms.CKA(
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
from MKLpy.algorithms import CKA
mkl = CKA().fit(KLtr, Ytr)
```

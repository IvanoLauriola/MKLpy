

# MKLpy








**MKLpy** is a framework for **M**ultiple **K**ernel **L**earning (MKL)  inspired by the [scikit-learn](http://scikit-learn.org/stable) project.
The library encapsulates *everything you need* to run MKL algorithms, from the kernels computation to the final evaluation.

<img src="./resources/mklpy_logo.svg" class="center">


MKLpy contains:

* the implementation of MKL algorithms (**EasyMKL**, **GRAM**);
* kernel functions (**polynomial**, **boolean** kernels, and **string** kernels);
* various metrics and tools (kernel_alignment, radius, margin, spectral ratio...)


The main MKL algorithms implemented in this library are

|Name       |Short description | Status | Source |
|-----------|------------------|--------|:------:|
| AverageMKL| Computes the simple average of base kernels         | Available | - |
| EasyMKL   | Fast and memory efficient margin-based combination  | Available |[[1]](https://www.sciencedirect.com/science/article/abs/pii/S0925231215003653) |
| GRAM      | Radius/margin ratio optimization                    | Available |[[2]](https://www.researchgate.net/publication/318468451_Radius-Margin_Ratio_Optimization_for_Dot-Product_Boolean_Kernel_Learning)   |
| R-MKL     | Radius/margin ratio optimization                    | Available |[[3]](https://link.springer.com/content/pdf/10.1007/978-3-642-04180-8_39.pdf)  |
| MEMO      | Margin maximization and complexity minimization     | Available |[[4]](https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es2018-181.pdf) |
| SimpleMKL | Alternate margin maximization                       | Work in progress |[[5]](http://www.jmlr.org/papers/volume9/rakotomamonjy08a/rakotomamonjy08a.pdf)|


!!! see
	The ```Tutorials``` section contains useful snippets of code to start with MKLpy.



- - -

## Installation

MKLpy is available on **PyPI**:
```sh
pip install MKLpy
```

MKLpy requires [pytorch](https://pytorch.org/), [scikit-learn](https://scikit-learn.org/stable/), and [cvxopt](https://cvxopt.org/) installed.




- - -

## Work in progress

MKLpy is under development! We are working to integrate several new features, including:

* further MKL algorithms, such as MEMO, and SimpleMKL;
* additional kernels for structured data (graphs, trees);
* efficient optimization

- - -

## Known issues

* When dealing with normalized kernels, you need to (i) compute the complete kernel matrix (training + test examples) and to (ii) split the matrix into training and test matrices. Currently, you cannot directly compute the normalized kernel for training and test. This is not efficient and it will be fixed in the next releases.

* Some boolean kernels (DNF and CNF kernels) are currently disabled.

* The documentation for developers, containing directives and tools to develop novel algorithms and functionalities, is currently not available.

* We're fixing some issues related to SimpleMKL. The algorithm will be available in the next weeks.


- - -

## Citing MKLpy

If you use MKLpy for a scientific purpose, please **cite** the following preprint.

```
@article{lauriola2020mklpy,
  title={MKLpy: a python-based framework for Multiple Kernel Learning},
  author={Lauriola, Ivano and Aiolli, Fabio},
  journal={arXiv preprint arXiv:2007.09982},
  year={2020}
}
```
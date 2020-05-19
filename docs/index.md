# MKLpy


**MKLpy** is a framework for **M**ultiple **K**ernel **L**earning (MKL)  inspired by the [scikit-learn](http://scikit-learn.org/stable) project.
The library encapsulates *everything you need* to run MKL algorithms, from the kernels computation to the final evaluation.


MKLpy contains:

* the implementation of MKL algorithms (**EasyMKL**, **GRAM**);
* kernel functions (**polynomial**, **boolean** kernels, and **string** kernels);
* various metrics and tools (kernel_alignment, radius, margin, spectral ratio...)

The ```examples``` section contains useful snippets of code.


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



!!! warning
	If you use MKLpy for a scientific purpose, please **cite** this library.
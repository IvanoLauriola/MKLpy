MKLpy
=====

[![Documentation Status](https://readthedocs.org/projects/mklpy/badge/?version=latest)](https://mklpy.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://travis-ci.com/IvanoLauriola/MKLpy.svg?branch=master)](https://travis-ci.com/IvanoLauriola/MKLpy)
[![Coverage Status](https://coveralls.io/repos/github/IvanoLauriola/MKLpy/badge.svg?branch=master&service=github)](https://coveralls.io/github/IvanoLauriola/MKLpy?branch=master&service=github)
[![PyPI version](https://badge.fury.io/py/MKLpy.svg)](https://badge.fury.io/py/MKLpy)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)


**MKLpy** is a framework for Multiple Kernel Learning (MKL)  inspired by the [scikit-learn](http://scikit-learn.org/stable) project.

This package contains:
* the implementation of some MKL algorithms;
* tools to operate on kernels, such as normalization, centering, summation, average...;
* metrics, such as kernel_alignment, radius of Minimum Enclosing Ball, margin between classes, spectral ratio...;
* kernel functions, including boolean kernels (disjunctive, conjunctive, DNF, CNF) and string kernels (spectrum, fixed length and all subsequences).


The main algorithms implemented in this library are
|Name      | Short description | Source |
|----------|-------------|--------|
| AverageMKL | Computes the simple average of base kernels | - |
| EasyMKL  | Fast and memory efficient algorithm maximizing an approximation of the margin | Aiolli et al. [1](https://www.sciencedirect.com/science/article/abs/pii/S0925231215003653) |
| GRAM     | Radius/margin optimization of the combined kernel | Lauriola et al. [2](https://www.researchgate.net/profile/Ivano_Lauriola/publication/318468451_Radius-Margin_Ratio_Optimization_for_Dot-Product_Boolean_Kernel_Learning/links/59c39754a6fdcc69b9354011/Radius-Margin-Ratio-Optimization-for-Dot-Product-Boolean-Kernel-Learning.pdf) |



The documentation of MKLpy is available on [readthedocs.io](https://mklpy.readthedocs.io/en/latest/)!



Installation
------------

**MKLpy** is also available on PyPI:
```sh
pip install MKLpy
```

**MKLpy** leverages multiple scientific libraries, that are [numpy](https://www.numpy.org/), [scikit-learn](https://scikit-learn.org/stable/), [PyTorch](https://pytorch.org/), and [CVXOPT](https://cvxopt.org/).


Examples
--------
The folder *examples* contains several scripts and snippets of codes to show the potentialities of **MKLpy**. The examples show how to train a classifier, how to process data, and how to use kernel functions.

Additionally, you may read our [tutorials](https://mklpy.readthedocs.io/en/latest/)


Work in progress
----------------
**MKLpy** is under development! We are working to integrate several features, including:
* further MKL algorithms, such as GRAM, MEMO, and SimpleMKL;
* more kernels for structured data;
* efficient optimization




Citing MKLpy
------------
If you use MKLpy for a scientific purpose, please cite this library.

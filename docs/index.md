# MKLpy


**MKLpy** is a framework for **M**ultiple **K**ernel **L**earning (MKL)  inspired by the [scikit-learn](http://scikit-learn.org/stable) project.

This package contains:

* the implementation of MKL algorithms, such as **EasyMKL** and **GRAM**;
* kernel functions, such as **polynomial**, **boolean** kernels, and **string** kernels;
* various metrics, such as kernel_alignment, radius, margin, spectral ratio...

The ```examples``` section contains useful snippets of code.


- - -

### Installation

**MKLpy** is available on PyPI:
```sh
pip install MKLpy
```

To work properly, **MKLpy** requires [numpy](https://www.numpy.org/), [scikit-learn](https://scikit-learn.org/stable/), and [cvxopt](https://cvxopt.org/).



- - -

### Work in progress

**MKLpy** is under development! We are working to integrate several new features, including:

* further MKL algorithms, such as MEMO, and SimpleMKL;
* more kernels for structured data (graphs, trees);
* efficient optimization;
* [PyTorch](https://pytorch.org/) as backend**!**

- - -

### Citing MKLpy

If you use MKLpy for a scientific purpose, please cite this library.
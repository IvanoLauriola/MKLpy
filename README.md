MKLpy
=====


**MKLpy** is a framework for Multiple Kernel Learning (MKL)  inspired by the [scikit-learn](http://scikit-learn.org/stable) project.

This package contains:
* the implementation of some MKL algorithms, such as EasyMKL;
* tools to operate on kernels, such as normalization, centering, summation, average...;
* metrics, such as kernel_alignment, radius of Minimum Enclosing Ball, margin between classes, spectral ratio...;
* kernel functions, including boolean kernels (disjunctive, conjunctive, DNF, CNF) and string kernels (spectrum, fixed length and all subsequences).


The documentation of MKLpy is available on [readthedocs.io](https://mklpy.readthedocs.io/en/latest/)!



Installation
------------

**MKLpy** is also available on PyPI:
```sh
pip install MKLpy
```

To work properly, **MKLpy** requires:

| resource       | website |
| ------       | ------ |
| numpy        | [https://www.numpy.org/](https://www.numpy.org/) |
| scikit-learn | [https://scikit-learn.org/stable/](https://scikit-learn.org/stable/) |
| cvxopt       | [https://cvxopt.org/](https://cvxopt.org/) |


Examples
--------
The folder *examples* contains several scripts and snippets of codes to show the potentialities of **MKLpy**. The examples show how to train a classifier, how to process data, and how to use kernel functions.
Currently, we ware working for a complete documentation.



Work in progress
----------------
**MKLpy** is under development! We are working to integrate several features, including:
* further MKL algorithms, such as GRAM, MEMO, and SimpleMKL;
* more kernels for structured data;
* incremental generators of kernels;
* [tensorflow](https://www.tensorflow.org/) as backend **!**


Citing MKLpy
------------
If you use MKLpy for a scientific purpose, please cite this library.
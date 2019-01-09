"""
==========================================================
MKLpy: A framework for Multiple Kernel Learning for Python
==========================================================

MKLpy is a framework for Multiple Kernel Learning and Kernel Learning for python
based on numpy, scikit-learn and cvxopt.

Contents
--------
The main contents of MKLpy are:
* some MKL algorithms and kernel machines, such as EasyMKL and KOMD;
* a meta-MKL-classifier used in multiclass problems according to one-vs-one pattern;
* a meta-MKL-classifier for MKL algorithms based on heuristics;
* tools to generate and handle list of kernels in an efficient way;
* tools to operate over kernels, such as normalization, centering, summation, mean...;
* metrics, such as kernel_alignment, radius...;
* kernel functions, such as HPK and SSK.

Subpackages
-----------
MKLpy is divided into subpackages, using any of these requires an explicit import.

::
    algorithms          --- MKL algorithms and kernel machines, such as EasyMKL and KOMD
    lists               --- Kernel generators and items used to handle list of functions
    metrics             --- Metrics used to evaluate kernels, such as kernel alignment, radius of MEB, margin between classes
    metrics.pairwise    --- Kernel functions, such as HPK and SSK
    utils               --- Tools used to validate a kernel list, check the interface of a generic MKL algorithm and other useful stuff
    arrange.py          --- Functions that combine a list of kernels, such as summation, mean and product over kernels
    regularization      --- Functions used to perform some transphormation over kernels and samples matrices, such as normalization, centering, rescale, tracenormalization
    multiclass          --- Meta-estimator for MKL in multiclass context
    heuristics          --- Meta-estimator for heuristic combinations of kernels
    test                --- A various set of tests used to check the framework

"""


#import algorithms,lists,metrics,test,utils

__all__ = ['algorithms',
           'utils',
           'preprocessing']
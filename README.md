MKLpy
=====


MKLpy is a framework for Multiple Kernel Learning and kernel machines scikit-compliant.

This package contains:

* MKL algorithms
  * EasyMKL
  * Average of kernels
  * Soon available: GRAM, MEMO, SimpleMKL

* tools to operate over kernels, such as normalization, centering, summation, mean...;

* metrics, such as kernel_alignment, radius, margin, spectral ratio...;

* kernel functions, such as homogeneous polynomial and boolean kernels (disjunctive, conjunctive, DNF, CNF).

The 'examples' folder contains useful snippets of code.


For more informations about classification, kernels and predictors visit `Link scikit-learn <http://scikit-learn.org/stable/>`_


requirements
------------

To work properly, MKLpy requires:

* numpy

* scikit-learn

* cvxopt

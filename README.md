# MKLpy
MKLpy is a framework for Kernel Learning and Multiple Kernel Learning based on scikit-learn.
To work properly, it requires scikit-learn, numpy, and cvxopt installed on our system.

This package contains:
- some MKL algorithms, such as EasyMKL;
- some kernel machines, such as KOMD;
- tools used to make a list of kernel functions avoiding the memory-hog;
- tools to operate over kernels, such as centering, normalizations, kernel_alignment...;
- metrics based on margin, MEB, spectral ratio;
- tools used to generate kernels in efficient way;
- a meta-MKLclassifier for multiclass-problems according to one-vs-one pattern;
- a meta-MKLclassifier for heuristics kernel combinations.

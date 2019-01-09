#from distutils.core import setup, find_packages
from setuptools import setup, find_packages
import os



setup(
  name = 'MKLpy',
  packages = find_packages(exclude=['build', '_docs', 'templates']),
  version = '0.3',
  install_requires=[
        "numpy",
        "scipy",
        "cvxopt",
        "scikit-learn"
  ],
  license = "GNU General Public License v3.0",
  description = 'A package for Multiple Kernel Learning scikit-compliant',
  author = 'Lauriola Ivano',
  author_email = 'ivano.lauriola@phd.unipd.it',
  url = 'https://github.com/IvanoLauriola/MKLpy',
  download_url = 'https://github.com/IvanoLauriola/MKLpy',
  keywords = ['kernel', 'MKL', 'learning', 'multiple kernel learning', 'EasyMKL','SVM','boolean kernels'],
  classifiers = [
                 'Development Status :: 3 - Alpha',
                 'Programming Language :: Python :: 3.6',
                 'Topic :: Scientific/Engineering :: Artificial Intelligence',
                 'Topic :: Scientific/Engineering :: Mathematics',
                 'License :: OSI Approved :: MIT License',
                ],
  long_description=
'''

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



''',
)

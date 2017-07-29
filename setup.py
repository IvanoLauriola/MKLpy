#from distutils.core import setup, find_packages
from setuptools import setup, find_packages
import os



setup(
  name = 'MKLpy',
  packages = find_packages(exclude=['build', '_docs', 'templates']),
  version = '0.1.3.2c',
  install_requires=[
        "numpy",
        "scipy",
        "cvxopt",
        "scikit-learn"
  ],
  license = "MIT",
  description = 'A package for Multiple Kernel Learning scikit-compliant',
  author = 'Lauriola Ivano',
  author_email = 'ivanolauriola@gmail.com',
  url = 'https://github.com/IvanoLauriola/MKLpy',
  download_url = 'https://github.com/IvanoLauriola/MKLpy',
  keywords = ['kernel', 'MKL', 'learning', 'multiple kernel learning', 'EasyMKL','SVM'],
  classifiers = [
                 'Development Status :: 3 - Alpha',
                 'Programming Language :: Python :: 2.7',
                 'Topic :: Scientific/Engineering :: Artificial Intelligence',
                 'Topic :: Scientific/Engineering :: Mathematics',
                 'License :: OSI Approved :: MIT License',
                ],
  long_description=
'''
=====
MKLpy
=====


MKLpy is a framework for Multiple Kernel Learning and kernel machines scikit-compliant.

This package contains:

* some MKL algorithms and kernel machines, such as EasyMKL and KOMD;

* a meta-MKL-classifier used in multiclass problems according to one-vs-one pattern;

* a meta-MKL-classifier for MKL algorithms based on heuristics;

* tools to generate and handle list of kernels in an efficient way;

* tools to operate over kernels, such as normalization, centering, summation, mean...;

* metrics, such as kernel_alignment, radius...;

* kernel functions, such as HPK and boolean kernels (disjunctive, conjunctive, DNF, CNF).



For more informations about classification, kernels and predictors visit `Link scikit-learn <http://scikit-learn.org/stable/>`_


requirements
------------

To work properly, MKLpy requires:

* numpy

* scikit-learn

* cvxopt


examples
--------


**Generation phase**

It is possible to exploit some generators to make a list of kernels. In the following example we rescale and normalize data, then we create a list of 20 Homogeneous Polynomial Kernels with degrees 1..20

.. code-block:: python

    from MKLpy.lists import HPK_generator
    from MKLpy.regularization import rescale_01, normalization
    X = rescale_01(X)   # X must be a dense matrix!!!
    X = normalization(X)
    KL = HPK_generator(X).make_a_list(20).to_array()

It is possible to create any custom lists

.. code-block:: python

    KL = np.array([np.dot(X,X.T)**d for d in range(1,21)])
    

**Training phase**

A kernel list is used as input of MKL algorithms. The interface of a generic MKL algorithm is the same of all predictors in scikit-learn, with the difference that the .fit method can has a list of kernels as input instead a single one or a samples matrix.

.. code-block:: python

    from MKLpy.algorithms import EasyMKL
    clf = EasyMKL(lam=0.1, kernel='precomputed')
    clf = clf.fit(KL,Y)


it is also possible to learn a kernel combination with an MKL algorithm and fit the model using another kernel machine, such as an SVC

.. code-block:: python

    from sklearn.svm import SVC
    ker_matrix = EasyMKL(lam=0.1, kernel='precomputed').arrange_kernel(KL,Y)
    ker_matrix = np.array(ker_matrix)
    clf = SVC(C=2, kernel='precomputed').fit(ker_matrix,Y)


**Evaluation phase**


.. code-block:: python

    from MKLpy.model_selection import cv3
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedShuffleSplit
    train,test = StratifiedShuffleSplit(n_split=1).split(X,Y).next()
    tr,te = cv3(train,test,n_kernels)
    KL_tr = KL[tr]
    KL_te = KL[te]
    Y_tr  = Y[train]
    Y_te  = Y[test]
    clf = EasyMKL(kernel='precomputed').fit(KL_tr,Y_tr)
    y_score = clf.decision_function(KL_te)
    AUC = roc_auc_score(Y_te,y_score)


**Some useful stuff**

some metrics...

.. code-block:: python

    from MKLpy.metrics import radius,margin
    K = np.dot(X,X)**2
    rMEB = radius(K)   //rMEB is the radius of the closest hypersphere that contains the data
    m = margin(K,Y)     //m is the margin between the classes, it works only in binary context





''',
)

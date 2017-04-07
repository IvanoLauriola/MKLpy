#from distutils.core import setup, find_packages
from setuptools import setup, find_packages
import os



setup(
  name = 'MKLpy',
  packages = find_packages(exclude=['build', '_docs', 'templates']),
  version = '0.1.2.1',
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

* kernel functions, such as HPK and SSK.



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

How to generate a list of functions and a list of kernel matrices using HPK kernel over the training data X.

A list of kernel functions can be used if the memory is not enough to hold all kernel matrices, but often using an explicit list of kernel matrices is faster.


.. code-block:: python

    from MKLpy.lists import HPK_generator
    from MKLpy.regularization import rescale_01, normalization
    X = rescale_01(X)
    X = normalization(X)
    K_func_list = HPK_generator(X).make_a_list(20)
    K_mat_list = K_func_list.to_array()


**Training phase**

A kernel list is used as input of MKL algorithms. The interface of a generic MKL algorithm is the same of all predictors in scikit-learn, with the difference that
the .fit method can has a list of kernels as input instead a single one or a samples matrix.

.. code-block:: python

    from MKLpy.algorithms import EasyMKL
    clf = EasyMKL(lam=0.1, tracenorm=False, kernel='precomputed')
    clf = clf.fit(K_list,Y)     //K_func_list or K_mat_list


it is also possible to learn a kernel combination with an MKL algorithm and fit the model using another kernel machine, such as an SVC

.. code-block:: python

    from sklearn.svm import SVC
    ker_matrix = EasyMKL(lam=0.1, kernel='precomputed').arrange_kernel(K_list,Y)
    clf = SVC(C=2, kernel='precomputed').fit(ker_matrix,Y)


**Training with samples matrix**

Due to usability and user experience, it is possible to use a samples matrices as input, like an estimator in scikit-learn.

.. code-block:: python

    clf = EasyMKL(lam=0.1)
    clf = clf.fit(X,Y)


**Evaluation phase**

If we use a samples matrix as input in an MKL algorithm, then the decision_function method can has the test samples matrix. Instead, if we use a precomputed list of kernels, we need to calculate a test kernel matrices for each kernel in list.

.. code-block:: python

    from sklearn.metrics import roc_auc_score
    K_list_tr = HPK_generator(Xtr).make_a_list(20).to_array()
    K_list_te = HPK_generator(Xtr,Xte).make_a_list(20).to_array()
    clf = EasyMKL(kernel='precomputed').fit(K_list_tr,Ytr)
    y_score = EasyMKL.decision_function(K_list_te)
    AUC = roc_auc_score(Yte,y_score)


**Some useful stuff**

some metrics

.. code-block:: python

    from MKLpy.metrics.pairwise import HPK_kernel as HPK
    from MKLpy.metrics import radius,margin
    K = HPK(X, degree=4)
    rMEB = radius(K)   //rMEB is the radius of the closest hypersphere that contains the data
    m = margin(K,Y)     //m is the margin between the classes, it works only in binary context





''',
)

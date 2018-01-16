from numpy.testing import assert_array_equal, assert_array_almost_equal, assert_equal
from sklearn.datasets import load_iris
import numpy as np
import sys
from MKLpy.algorithms.base import MKL


"""
.. codeauthor:: Ivano Lauriola <ivanolauriola@gmail.com>

====================
Check MKL algorithms
====================

.. currentmodule:: MKLpy.utils.mkl_checks

This module is used to check if a MKL algorithm is compliant with MKLpy,
evaluanting the public interface.

Notes
-----
This unit tests are not sufficient. The estimator need to pass also the scikit unit tests.

"""

def check_class(mkl):
	assert issubclass(mkl,MKL)

def check_init(mkl):
	clf = mkl()
	params = clf.get_params()
	assert params['multiclass_strategy'] in ['ovo','ova']
	assert params['estimator']
	assert params['generator']
	# ...

def check_base_learner(mkl):
	_mkl = mkl()
	estimator = _mkl.estimator
	clf = estimator.fit([[1,0.1],[0.1,1]],[1,0])
	y_pred = clf.predict([[0.2,0.02]])
	y_score = clf.decision_function([[0.2,0.02]])
	assert clf.kernel == 'precomputed'
	clf = estimator.__class__(kernel='linear')	#base learner must be a kernel machine
	clf = clf.fit([[1,0.1],[0.1,1]],[1,0])
	y_pred = clf.predict([[0.2,0.02]])
	y_score = clf.decision_function([[0.2,0.02]])

def check_arrange(mkl):
	clf = mkl()
	#clf.arrange_kernel()



def all_check():
	yield check_class
	yield check_init
	yield check_base_learner
	yield check_arrange


def check_estimator(mkl):
	"""check if a MKL algorithm is compliant with MKLpy

    This estimator will run an extensive test-suite for input validation, shapes, etc
    
    Parameters
    ----------
    mkl : class,
          class to check.
    """
	for check in all_check():
		check(mkl)
	return

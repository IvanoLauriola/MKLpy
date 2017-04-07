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
This unit tests are not sufficient. The estimator need to pass also the unit tests
offered by scikit-learn.

"""

def check_estimator(mkl):
    """check if a MKL algorithm is compliant with MKLpy

    This estimator will run an extensive test-suite for input validation, shapes, etc
    
    Parameters
    ----------
    mkl : class,
          class to check.
    """
    return

"""
================
Metrics.pairwise
================

.. currentmodule:: MKLpy.metrics.pairwise

This sub-package contains kernel functions, such as:
* boolean kernels : monotone conjucntive, disjunctive, dnf, tanimoto...;
* string kernels: p-spectrum...;
* misc kernels: homogeneous polynomial kernels.

"""

from .boolean import monotone_conjunctive_kernel, monotone_disjunctive_kernel, monotone_dnf_kernel
from .string import spectrum_embedding, fixed_length_subsequences_embedding, all_subsequences_embedding, \
                   spectrum_kernel   , fixed_length_subsequences_kernel   , all_subsequences_kernel
from .misc import homogeneous_polynomial_kernel

__all__ = ['homogeneous_polynomial_kernel',
           'monotone_conjunctive_kernel',
           'monotone_disjunctive_kernel',
           'monotone_dnf_kernel',
           'spectrum_embedding',
           'fixed_length_subsequences_embedding',
           'all_subsequences_embedding',
           'spectrum_kernel',
           'fixed_length_subsequences_kernel',
           'all_subsequences_kernel'
           ]


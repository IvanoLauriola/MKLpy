"""
.. codeauthor:: Ivano Lauriola <ivanolauriola@gmail.com>

================
Kernel functions
================

.. currentmodule:: MKLpy.metrics.pairwise

This module contains kernel functions for strings and sequences.

"""

import numpy as np
from scipy.special import binom
from scipy.sparse import issparse
from MKLpy.utils.validation import check_X_T
import types
import itertools




def spectrum_embedding(X, p=2, binary=False):
	'''
	Computes the spectrum embedding of a seuqnce.
	The feature space contains the number of occurrences of all possible substrings

	p: the length of substructures
	binary: counts of occurrences or 1/0
	'''
	S = []
	for x in X:
		vx = {}
		for i in range(len(x)-p+1):
			u = x[i:i+p]
			vx[u] = 1 if u not in vx or binary else vx[u] + 1
		S.append(vx)
	return S



def fixed_length_subsequences_embedding(X, p=2, binary=False):
	S = []
	for x in X:
		vx = {}
		for u in itertools.combinations(x,p):
			vx[u] = 1 if u not in vx or binary else vx[u] + 1
		S.append(vx)
	return S


def all_subsequences_embedding(X, binary=False):
	S = []
	for x in X:
		vx = {():1}
		for p in range(1, len(x)+1):
			for u in itertools.combinations(x,p):
				vx[u] = 1 if u not in vx or binary else vx[u] + 1
		S.append(vx)
	return S



def dictionary_dot(EX, ET):
	K = np.zeros((len(ET), len(EX)))
	for it, vt in enumerate(ET):
		for ix, vx in enumerate(EX):
			K[it,ix] = sum( vx[f]*vt[f] for f in vt.keys() & vx.keys() )
	return K



def spectrum_kernel(X, T=None, p=2, binary=False):	
	X, T = check_X_T(X, T)
	EX = spectrum_embedding(X, p=p, binary=binary)
	ET = spectrum_embedding(T, p=p, binary=binary)
	return dictionary_dot(EX, ET)


def fixed_length_subsequences_kernel(X, T=None, p=2, binary=False):
	X, T = check_X_T(X, T)
	EX = fixed_length_subsequences_embedding(X, p=p, binary=binary)
	ET = fixed_length_subsequences_embedding(T, p=p, binary=binary)
	return dictionary_dot(EX, ET)


def all_subsequences_kernel(X, T=None, binary=False):
	X, T = check_X_T(X, T)
	EX = all_subsequences_embedding(X, binary=binary)
	ET = all_subsequences_embedding(T, binary=binary)
	return dictionary_dot(EX, ET)
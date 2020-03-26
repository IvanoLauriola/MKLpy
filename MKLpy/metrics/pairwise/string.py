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
from sklearn.metrics.pairwise import check_pairwise_arrays
import itertools
from multiprocessing import Pool, cpu_count
import joblib
from threading import Thread
from sklearn.utils import parallel_backend as pb




def spectrum_embedding(x, p=2, binary=False):
	'''
	Computes the spectrum embedding of a seuqnce.
	The feature space contains the number of occurrences of all possible substrings

	p: the length of substructures
	binary: counts of occurrences or 1/0
	'''
	vx = {}
	for i in range(len(x)-p+1):
		u = x[i:i+p]
		vx[u] = 1 if u not in vx or binary else vx[u] + 1
	return vx


def fixed_length_subsequences_embedding(x, p=2, binary=False):
	vx = {}
	for u in itertools.combinations(x,p):
		vx[u] = 1 if u not in vx or binary else vx[u] + 1
	return vx


def all_subsequences_embedding(x, binary=False):
	vx = {():1}
	for p in range(1, len(x)+1):
		for u in itertools.combinations(x,p):
			vx[u] = 1 if u not in vx or binary else vx[u] + 1
	return vx




def spectrum_kernel(X, Z=None, p=2, binary=False):	
	embedding = lambda s : spectrum_embedding(s, p=p, binary=binary)
	return string_kernel(embedding, X=X, Z=Z)


def fixed_length_subsequences_kernel(X, Z=None, p=2, binary=False):
	embedding = lambda s : fixed_length_subsequences_embedding(s, p=p, binary=binary)
	return string_kernel(embedding, X=X, Z=Z)
	

def all_subsequences_kernel(X, Z=None, binary=False):
	embedding = lambda s : all_subsequences_embedding(s, binary=binary)
	return string_kernel(embedding, X=X, Z=Z)




def string_kernel(embedding, X, Z=None):
	EX = [embedding(s) for s in X]
	EZ = EX if (Z is None) or (Z is X) else [embedding(s) for s in Z]
	return dictionary_dot(EX, EZ)


def dictionary_dot(EX, EZ):
	K = np.zeros((len(EX), len(EZ)))
	for iz, vz in enumerate(EZ):
		for ix, vx in enumerate(EX):
			K[ix,iz] = sum( vx[f]*vz[f] for f in vz.keys() & vx.keys() )
	return K

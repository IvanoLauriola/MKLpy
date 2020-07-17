# -*- coding: latin-1 -*-

"""
@author: Ivano Lauriola
@email: ivano.lauriola@phd.unipd.it, ivanolauriola@gmail.com

This file is part of MKLpy: a scikit-compliant framework for Multiple Kernel Learning
This file is distributed with the GNU General Public License v3 <http://www.gnu.org/licenses/>.  

"""

from .data_preprocessing import \
    normalization,             \
    rescale,                   \
    rescale_01,                \
    centering

from .kernel_preprocessing import \
    kernel_normalization,        \
    tracenorm,                   \
    kernel_centering


__all__ = ['kernel_normalization',
           'tracenorm'
           'kernel_centering',
           'normalization',
           'rescale',
           'rescale_01',
           'centering',
           ]


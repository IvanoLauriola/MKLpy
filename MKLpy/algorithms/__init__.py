# -*- coding: latin-1 -*-

"""
@author: Ivano Lauriola
@email: ivano.lauriola@phd.unipd.it

This file is part of MKLpy: a scikit-compliant framework for Multiple Kernel Learning
This file is distributed with the GNU General Public License v3 <http://www.gnu.org/licenses/>.  

"""


from .base import MKL
from .AverageMKL import AverageMKL
from .EasyMKL import EasyMKL
from .GRAM import GRAM
from .komd import KOMD
from .MEMO import MEMO

__all__ = [
    'AlternateMKL',
    'AverageMKL',
    'MKL',
    'EasyMKL',
    'GRAM',
    'KOMD',
    'MEMO',
]

# -*- coding: latin-1 -*-

"""
@author: Ivano Lauriola
@email: ivano.lauriola@phd.unipd.it

This file is part of MKLpy: a scikit-compliant framework for Multiple Kernel Learning
This file is distributed with the GNU General Public License v3 <http://www.gnu.org/licenses/>.  

"""

from .base import MKL
from .AverageMKL import AverageMKL
from .AlternateMKL import AlternateMKL
from .komd import KOMD
from .EasyMKL import EasyMKL
from .MEMO import MEMO
#from HeuristicMKLClassifier import HeuristicMKLClassifier

__all__ = ['EasyMKL',
           'KOMD',
           'MKL',
           'AverageMKL',
           'AlternateMKL',
           'MEMO'
           ]

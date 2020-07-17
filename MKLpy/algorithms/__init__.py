# -*- coding: latin-1 -*-

"""
@author: Ivano Lauriola
@email: ivano.lauriola@phd.unipd.it, ivanolauriola@gmail.com

This file is part of MKLpy: a scikit-compliant framework for Multiple Kernel Learning
This file is distributed with the GNU General Public License v3 <http://www.gnu.org/licenses/>.  

"""


from .base import MKL, OneStepMKL, TwoStepMKL, Cache, Solution
from .AverageMKL import AverageMKL
from .EasyMKL import EasyMKL
from .GRAM import GRAM
from .komd import KOMD
from .MEMO import MEMO
#from .SimpleMKL import SimpleMKL	available soon
from .RMKL import RMKL

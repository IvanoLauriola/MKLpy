"""
=======
Metrics
=======

.. currentmodule:: MKLpy.metrics

This sub-package contains metrics used to evaluate and compare kernels.
In particular, exists 3 kind of metrics:
* evaluation : let a kernel matrix, return a numerical value;
* similarity : let two kernel matrices, returns a similarity value between them;
* pairwise : let a set of samples, returns the corresponding kernel matrix.

This package contains the first and second kind of metrics,
the latter is implemented in a sub-package

Subpackages
-----------

Metrics contains as sub-package:

::
    pairwise        --- Kernel functions

"""

from .evaluate  import *
from .alignment import *


__all__ = ['radius',
           'margin',
           'ratio',
           'alignment',
           'alignment_ID',
           'alignment_yy',
           'trace',
           'frobenius',
           'spectral_ratio',
           ]

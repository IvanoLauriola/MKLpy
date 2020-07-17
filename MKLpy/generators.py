# -*- coding: latin-1 -*-

"""
@author: Ivano Lauriola
@email: ivano.lauriola@phd.unipd.it, ivanolauriola@gmail.com

This file is part of MKLpy: a scikit-compliant framework for Multiple Kernel Learning
This file is distributed with the GNU General Public License v3 <http://www.gnu.org/licenses/>.  

"""

from .metrics import pairwise
from .utils.misc import identity_kernel
from .utils.validation import check_pairwise_X_Z
import torch




class Generator:

    n_kernels = None

    def __init__(self, X, Z=None, include_identity=False):
        self.X, self.Z = check_pairwise_X_Z(X, Z)
        self.include_identity = include_identity

    def __len__(self):
        return self.n_kernels + self.include_identity

    def __getitem__(self, r):
        if r >= len(self):
            raise IndexError('List index (%d) out of range' % r)
        if self.include_identity and r == len(self)-1:
            if self.X is self.Z:
                K = identity_kernel(len(self.X)) 
            else:
                K = torch.zeros(len(self.X), len(self.Z))
        else:
            K = self._get_kernel(r)
        return K

    def _get_kernel(self, r):
    	raise NotImplementedError('This method has to be implemented in the derived class')

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx >= len(self):
            raise StopIteration
        self.idx += 1
        return self[self.idx-1]

    def to_list(self):
        return [K for K in self]



class Multiview_generator(Generator):
    
    def __init__(self, XL, ZL=None, include_identity=False, kernel=pairwise.linear_kernel):
        self.XL = XL
        self.ZL = XL if (ZL is None) or (ZL is XL) else ZL
        super().__init__(X=self.XL[0], Z=self.ZL[0], include_identity=include_identity)
        self.kernel = kernel
        self.n_kernels = len(XL)

    def _get_kernel(self, r):
        return self.kernel(self.XL[r], self.ZL[r])



class Lambda_generator(Generator):

    def __init__(self, X, Z=None, include_identity=False, kernels=[]):
        super().__init__(X=X, Z=Z, include_identity=include_identity)
        self.kernels   = kernels
        self.n_kernels = len(kernels)

    def _get_kernel(self, r):
        return self.kernels[r](self.X, self.Z)



class HPK_generator(Generator):

    def __init__(self, X, Z=None, include_identity=False, degrees=range(1,11), cache=True):
        super().__init__(X=X, Z=Z, include_identity=include_identity)
        self.degrees   = degrees
        self.cache     = cache
        self.n_kernels = len(degrees)
        if self.cache:
            self.L = pairwise.linear_kernel(self.X, self.Z)

    def _get_kernel(self, r):
        L = self.L if self.cache else pairwise.linear_kernel(self.X, self.Z)
        return L**self.degrees[r]



class RBF_generator(Generator):

    def __init__(self, X, Z=None, include_identity=False, gamma=[0.01, 0.1, 1], cache=True):
        super().__init__(X=X, Z=Z, include_identity=include_identity)
        self.gamma   = gamma
        self.cache     = cache
        self.n_kernels = len(gamma)
        if self.cache:
            self.D = pairwise.euclidean_distances(self.X, self.Z)**2

    def _get_kernel(self, r):
        D = self.D if self.cache else pairwise .euclidean_distances(self.X, self.Z)**2
        return torch.exp(-self.gamma[r] * D)




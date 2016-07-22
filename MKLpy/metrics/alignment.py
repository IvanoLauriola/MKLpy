
from pairwise import identity_kernel, ideal_kernel
#from MKLpy.regularization import kernel_centering

'''SOLO TIPI DI KERNEL_ALIGNMENT'''

def alignment (K1, K2):
    def ff(k1,k2):
        n = len(k1)
        s = 0
        for i in xrange(n):
            for j in xrange(n):
                s += k1[i,j]*k2[i,j]
        return s
    f0 = ff(K1,K2)
    f1 = ff(K1,K1)
    f2 = ff(K2,K2)
    return (f0 / sqrt(f1*f2))

def alignment_ID(K):
    return alignment(K,identity_kernel(K))

def alignment_yy(K,y1,y2=None):
    return alignment(K,ideal_kernel(y1,y2))

#def centered_alignment(K1,K2):
#    C1 = kernel_centering(K1.copy())
#    C2 = kernel_centering(K2.copy())
#    return alignment(C1,C2)

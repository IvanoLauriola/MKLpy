'''

This is a snippet of code showing how to use string kernels

Author: Ivano Lauriola, ivano.lauriola@phd.unipd.it

'''


# import string kernels
from MKLpy.metrics.pairwise import spectrum_kernel, fixed_length_subsequences_kernel, all_subsequences_kernel

# the training set consists of a list of strings
X = ['barasea','baads','careaw','catda','asdasd','daxsd','gfd','gfds','sadca','asdc']


# compute the p-spectrum kernel
Kp = spectrum_kernel(X, p=2)

# compute the fixed_length_subsequences_kernel
Kf = fixed_length_subsequences_kernel(X, p=2)

# compute the all_subsequences_kernel
Ka = all_subsequences_kernel(X)	# N.B.: no hyper-parameters are required



# if you are interested on the explicit representation, you may call the embedding functions
from MKLpy.metrics.pairwise import spectrum_embedding, fixed_length_subsequences_embedding, all_subsequences_embedding

# string embeddings are sparse by definition, hence, we adopt a dict-based notation
Ep = spectrum_embedding(X, p=2)
Ef = fixed_length_subsequences_embedding(X, p=2)
Ea = all_subsequences_embedding(X)

print(Ep)


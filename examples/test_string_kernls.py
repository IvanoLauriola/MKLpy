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



'''	what about relaxed embeddings?
	A 'relaxed' embedding is obtained by grouping set of characters in a single one.
	This process affects the sparsity of the representation.
'''
from MKLpy.preprocessing.relaxation import sequences_relaxation

print ('example of base relaxation')
X = ['asdASD','0123qwerty','Test','HH(1)']
relaxed_strings = sequences_relaxation(X)	# here we use the standard relaxation
for s in zip(X, relaxed_strings):
	print ('\'%20s\' -> \'%s\'' % s)


# how to define a custom relaxation?
rmap = {'a': 'asdqw', '0': '0123', 'A': 'THAS'} # each group of characters (values) is associated to a specific symbol (key)
print ('example of custom relaxation')
relaxed_strings = sequences_relaxation(X, relaxation=rmap)	# here we use the standard relaxation
for s in zip(X, relaxed_strings):
	print ('\'%20s\' -> \'%s\'' % s)

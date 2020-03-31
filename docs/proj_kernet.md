## Kernet


This page contains a short description of KerNET, an ensemble system that combines neural networks and MKL. 

In short, KerNET consists of two components:

* a neural network that learns and extracts base (or weak) kernels;
* a MKL algorithm that combines the base kernels and performs classification.

!!! Paper
	An exhaustive description of KerNET is available in the following paper:<br>
	Ivano Lauriola, Claudio Gallicchio, and Fabio Aiolli: *"Enhancing deep neural networks via Multiple Kernel Learning". Pattern Recognition (2020)*

The second part of the ensemble can be easily implemented with MKLpy, whereas the first part, i.e. the training of the neural network, depends on external libraries, such as [Tensorflow](https://www.tensorflow.org/), [Pytorch](https://pytorch.org/), or [Keras](https://keras.io/).

We show in the following a complete example of KerNET when using Keras as neural networks library. 
In the example, we use a simple multilayer feed forward dense neural network for simplicity. However, the same approach can be applied to any possible neural network.

```python
from keras.models import Sequential, Model
from keras.layers import Dense
from keras import callbacks as callb
from keras.utils import to_categorical

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from MKLpy.algorithms import EasyMKL
from MKLpy.utils.misc import identity_kernel
from MKLpy.preprocessing import normalization


data = load_iris()	#let's try with iris dataset!
X,Y = data.data, data.target
num_classes = len(np.unique(Y))

Yh = to_categorical(Y)	#I need one-hotted labels for training the NN
Xtr, Xva, Ytr, Yva, Ytr_1h, Yva_1h = train_test_split(X, Y,
        Yh, random_state=42, shuffle=True, test_size=.3)

num_classes = len(np.unique(Y))

#parameters of the network
learning_rate = 1e-5
batch_size    = 32
activation    = 'sigmoid'
num_hidden    = 10	 #num hidden layers
num_neurons   = 128  #num neurons per layer
max_epochs    = 100

#model setting
model = Sequential()
for l in range(1, num_hidden+1):	#add hidden layers
	layer = Dense(num_neurons, activation=activation, name='hidden_%d' % l)
	model.add(layer)
classification_layer = Dense(num_classes, activation='softmax', name='output')
model.add(classification_layer)

#optional equipments
reduce_lr  = callb.ReduceLROnPlateau(
	monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
earlystop  = callb.EarlyStopping(
	monitor='val_loss',patience=10, mode='min',verbose=1)

#compilation
model.compile(optimizer='sgd', loss='mse', metrics=['accuracy'])
history = model.fit(
	Xtr, Ytr_1h, 
	batch_size=batch_size, 
	epochs=max_epochs, 
	validation_data=(Xva, Yva_1h), 
	verbose=1,
	callbacks=[reduce_lr,earlystop])

#representations extraction and kernels definition
def extract_reps(X, net):	
	''' this function extracts intermediate representations
		developed by network for each input example in X.
	'''
	for l in range(1, num_hidden+1):
		representations = []
		layer_name = 'hidden_%d' % l
		partial_model = Model(
			inputs=model.input, 
			outputs=model.get_layer(layer_name).output)
		#rep_l contains the representation developed at the 
		#layer l for each input examples
		rep_l = partial_model.predict(X).astype(np.double)
		rep_l = normalization(rep_l)	#we always prefer to normalize data
		representations.append(rep_l)
	return representations

# here, XL is a list containing matrices
# the i-th matrix contains the representations for all input examples
# developed at the i-th hidden layer
XLtr = extract_reps(Xtr, model)
XLva = extract_reps(Xva, model)

# now, we can create our kernels list
# in this specific case, we compute linear kernels
KLtr = [X   @ X.T for X in XLtr]
KLva = [Xva @ X.T for Xva, X in zip(XLva, XLtr)]

# have you seen the section *best practices* ?
# I just ass the base input rerpesentation and an identity matrix
KLtr += [Xtr @ Xtr.T, identity_kernel(len(Ytr))]
KLva += [Xva @ Xtr.T, np.zeros(KLva[0].shape)]

# MKL part
mkl = EasyMKL().fit(KLtr, Ytr)
y_preds = mkl.predict(KLva)

# final evaluation
accuracy = accuracy_score(Yva, y_preds)
print (accuracy)
```
# -*- coding: utf-8 -*-
"""
.. codeauthor:: Michele Donini <>
.. codeauthor:: Ivano Lauriola <ivanolauriola@gmail.com>



"""

import numpy as np
import sys
from cvxopt import matrix
from sklearn.metrics.pairwise import polynomial_kernel
from sklearn import svm
#from utils import HPK_generator
from sklearn.metrics import roc_auc_score, accuracy_score

class OneVsOneMKLClassifier():
    
    def __init__(self, clf1, verbose=False):
        #print 'init ovo'
        self.clf1 = clf1
        self.clf2 = clf1.estimator
        self.verbose = verbose
        
    def get_params(self, deep=True):
        return {}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self,parameter,value)
        return self
    
    def fit(self,K_tr,Y_tr):
        self.K_tr = K_tr
        id_for_train = []
        for l in set(Y_tr):
            id_for_train += [idx for idx,lab in enumerate(Y_tr) if lab==l]
        #id_for_train = [i for l in Y_tr]
        
        #ordering che devo ancora capì
        K_tr = np.array([kk[id_for_train][:,id_for_train] for kk in K_tr])
        
        n_classes = len(np.unique(Y_tr))
        self.classes_ = np.unique(Y_tr)
        self.n_classes = n_classes
        #Ove vs One
        list_of_dichos = []
        for i in range(n_classes):
            for j in range(i+1,n_classes):
                #list_of_dichos.append(((i,),(j,)))
                list_of_dichos.append(((int(self.classes_[i]),),(int(self.classes_[j]),)))
        
        list_of_indices = {}
        list_of_indices_train = {}
        list_of_labels = {}
        list_of_labels_train = {}
        #prendo gli esempi per ogni dicotomia
        for dicho in list_of_dichos:
            list_of_indices[dicho] = [[i for i,l in enumerate(Y_tr) if l in dicho[0]],
                                      [i for i,l in enumerate(Y_tr) if l in dicho[1]]]
            list_of_indices_train[dicho] = [[i for i,l in enumerate(id_for_train) if Y_tr[l] in dicho[0]],
                                            [i for i,l in enumerate(id_for_train) if Y_tr[l] in dicho[1]]]
            
            list_of_labels[dicho] = [1.0]*len(list_of_indices[dicho][0]) + [-1.0]*len(list_of_indices[dicho][1])
            list_of_labels_train[dicho] = [1.0]*len(list_of_indices_train[dicho][0]) + [-1.0]*len(list_of_indices_train[dicho][1])
            
        if self.verbose:
            print ('Learning the models for %d dichotomies' % len(list_of_dichos))
        # LEARNING THE MODELS
        wmodels = {}
        combinations = {}
        functional_form = self.clf1.how_to
        


        for dicho in list_of_dichos:
            ind = list_of_indices_train[dicho][0] + list_of_indices_train[dicho][1]
            cc = self.clf1.__class__(**self.clf1.get_params())
            cc.kernel = 'precomputed'
            #cc = cc.fit(np.array([kk[ind][:,ind]  for kk in K_tr]),
            ker_matrix = cc.arrange_kernel(np.array([kk[ind][:,ind]  for kk in K_tr]),
                       np.array(list_of_labels_train[dicho]))
            wmodels[dicho] = [w / sum(cc.weights) for w in cc.weights]
            combinations[dicho] = ker_matrix#cc.ker_matrix

            del cc
        
        self.ker_matrices = combinations #mi serve solo per alcuni test
        # Train SVM
        if self.verbose:
            print ('SVM training phase...')
        
        svcs = {}
        for dicho in list_of_dichos:
            svcs[dicho] = self.clf2.__class__(**self.clf2.get_params())
            svcs[dicho].kernel = 'precomputed'
            idx = list_of_indices[dicho][0]+list_of_indices[dicho][1]
            k = combinations[dicho]#[idx,:][:,idx]
            svcs[dicho].fit(np.array(k), np.array(list_of_labels[dicho]))
            #print svcs[dicho].n_support_
            #raw_input('b')
            #if self.verbose:
            #    sys.stdout.flush()
                
        #salvo gli oggetti che mi torneranno utili
        self.list_of_dichos = list_of_dichos
        self.svcs = svcs
        self.id_for_train = id_for_train
        self.list_of_indices = list_of_indices
        self.wmodels = wmodels
        self.weights = wmodels
        self.functional_form = functional_form
        return self



    def predict(self,K_te):#, Y_te):
        predicts = {}
        wmodels = self.wmodels
        single_accuracy = {}

        for dicho in self.list_of_dichos:
            idx = self.list_of_indices[dicho][0]+self.list_of_indices[dicho][1]
            w = self.wmodels[dicho]
            k = self.functional_form(np.array([kk[:,idx] for kk in K_te]),w)
            predicts[dicho] = self.svcs[dicho].decision_function(k)

        # Voting   
        #nn = len(Y_te)
        nn = len(K_te[0])
        #points = np.zeros((nn,self.n_classes),dtype=int)
        points = np.zeros((nn,int(np.max(self.classes_))+1),dtype=int)
        #print points.shape
        for dicho in self.list_of_dichos:
            for ir,r in enumerate(predicts[dicho]):
                if r > 0:
                    points[ir,dicho[0][0]] += 1
                else:
                    points[ir,dicho[1][0]] += 1

        y_pred = np.argmax(points,1)
        sys.stdout.flush()
        return y_pred


class OneVsRestMKLClassifier():
    def __init__(self, clf1, verbose=False):
        self.clf1 = clf1
        self.clf2 = clf1.estimator
        self.verbose = verbose

        self.is_fitted = False
        self.classes_ = None

    def fit(self, K_list, Y):
        n = len(Y)
        classes_ = np.unique(Y)
        n_classes = len(classes_)
        labels = {}
        for l in classes_:
            labels.update({l: [1 if _y == l else -1 for _y in Y]})
        weights = {}
        clfs = {}
        ker_matrices = {}

        # learning the models
        for model in classes_:
            # print 'learning model with ',model,' is the positive class'
            # learning the kernel
            cc1 = self.clf1.__class__(**self.clf1.get_params())
            cc1.kernel = 'precomputed'
            ker_matrix = cc1.arrange_kernel(K_list, labels[model])
            weights.update({model: cc1.weights})

            # fitting the model
            cc2 = self.clf2.__class__(**self.clf2.get_params())
            cc2.kernel = 'precomputed'
            cc2.fit(ker_matrix, labels[model])
            clfs.update({model: cc2})
            ker_matrices.update({model:ker_matrix})
            # del ker_matrix, cc1

        # save stuff
        self.classes_ = classes_
        self.functional_form = self.clf1.how_to
        # self.functional_form = lambda X,Y : self.clf1.how_to
        self.weights = weights
        self.clfs = clfs
        self.n_classes = n_classes
        self.classes_ = classes_
        self.ker_matrices = ker_matrices
        self.is_fitted = True
        return self

    def predict(self, K_list):
        if not self.is_fitted:
            raise Exception('huehuehue')
        # predict with binary models
        predicts = {}
        nn = K_list[0].shape[0]
        for model in self.classes_:
            w = self.weights[model]
            functional_form = self.functional_form
            ker = functional_form(K_list, w)
            clf = self.clfs[model]
            predicts.update({model: clf.decision_function(ker)})
        # voting
        scoring = np.zeros((nn, self.n_classes))
        for col, model in enumerate(self.classes_):
            scoring[:, col] = predicts[model]
        y_pred = np.array([self.classes_[np.argmax(sc)] for sc in scoring])
        return y_pred



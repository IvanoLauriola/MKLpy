# -*- coding: utf-8 -*-
"""
@author: Michele Donini, Lauriola Ivano

"""

import numpy as np
import sys
from cvxopt import matrix
from sklearn.metrics.pairwise import polynomial_kernel
from sklearn import svm
#from utils import HPK_generator
from sklearn.metrics import accuracy_score

class OneVsOneMKLClassifier():
    
    def __init__(self, clf1, clf2, verbose=False):
        self.clf1 = clf1
        self.clf2 = clf2
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
        self.n_classes = n_classes
        #Ove vs One
        list_of_dichos = []
        for i in range(n_classes):
            for j in range(i+1,n_classes):
                list_of_dichos.append(((i,),(j,)))
        
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
                print dicho[0],'vs',dicho[1],'->',len(list_of_indices[dicho][0]),'+1 vs',len(list_of_indices[dicho][1]),'-1'
                sys.stdout.flush()

        if self.verbose:
            print 'Learning the models for the',len(list_of_dichos),'dichotomies'
        # LEARNING THE MODELS
        wmodels = {}
        combinations = {}
        functional_form = self.clf1.how_to()
        
        for dicho in list_of_dichos:
            ind = list_of_indices_train[dicho][0] + list_of_indices_train[dicho][1]
            cc = self.clf1.__class__(**self.clf1.get_params())
            cc.kernel = 'precomputed'
            cc = cc.fit(np.array([kk[ind][:,ind]  for kk in K_tr]),
                       np.array(list_of_labels_train[dicho]))
            wmodels[dicho] = [w / sum(cc.weights) for w in cc.weights]#da sistemare per astrarre
            if self.verbose:
                print 'Model generated for the dichotomy:',dicho[0],'vs',dicho[1]
                print 'Weights:',len(cc.weights),sum(cc.weights),cc.weights
                sys.stdout.flush()
            combinations[dicho] = cc.ker_matrix
            del cc
        
        # RELEASE MEMORY
        #del k_list

        # Train SVM
        if self.verbose:
            print 'SVM training phase...'
        
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
            predicts[dicho] = self.svcs[dicho].predict(k)
            #print predicts[dicho][:20]
            #single_accuracy[dicho] = accuracy_score([1.0 if i in dicho[0] else -1.0 for i in Y_te], list(predicts[dicho]))
            #if self.verbose:
            #    print 'Accuracy test:',single_accuracy[dicho]
            #    print accuracy_score([1.0 if i in dicho[0] else -1.0 for i in Y_te], list(predicts[dicho]), normalize=False),'/',len(Y_te)
            
        

        # Voting   
        #nn = len(Y_te)
        nn = len(K_te[0])
        points = np.zeros((nn,self.n_classes),dtype=int)
        for dicho in self.list_of_dichos:
            for ir,r in enumerate(predicts[dicho]):
                if r > 0:
                    points[ir,dicho[0][0]] += 1
                else:
                    points[ir,dicho[1][0]] += 1

        y_pred = np.argmax(points,1)
        sys.stdout.flush()
        return y_pred
        
        
        

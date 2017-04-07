import numpy as np
import threading

class OneVsRestMKLClassifier():

    def __init__(self, clf1, clf2, verbose=False):
        self.clf1 = clf1
        self.clf2 = clf2
        self.verbose = verbose

        self.is_fitted = False
        self.classes_ = None

    def fit(self, K_list, Y):
        n = len(Y)
        classes_ = np.unique(Y)
        n_classes = len(classes_)
        labels = {}
        for l in classes_:
            labels.update({l : [1 if _y==l else -1 for _y in Y]})
        weights = {}
        clfs = {}
        
        #learning the models
        for model in classes_:
            #print 'learning model with ',model,' is the positive class'
            #learning the kernel
            cc1 = self.clf1.__class__(**self.clf1.get_params())
            cc1.kernel = 'precomputed'
            ker_matrix = cc1.arrange_kernel(K_list, labels[model])
            weights.update({model : cc1.weights})

            #fitting the model
            cc2 = self.clf2.__class__(**self.clf2.get_params())
            cc2.kernel = 'precomputed'
            cc2.fit(ker_matrix, labels[model])
            clfs.update({model : cc2})
            del ker_matrix, cc1

        #save stuff
        self.classes_ = classes_
        self.functional_form = self.clf1.how_to
        #self.functional_form = lambda X,Y : self.clf1.how_to
        self.weights = weights
        self.clfs = clfs
        self.n_classes = n_classes
        self.classes_ = classes_
        self.is_fitted = True
        return self

    def predict(self, K_list):
        if not self.is_fitted:
            raise Exception('huehuehue')
        #predict with binary models
        predicts = {}
        nn = K_list[0].shape[0]
        for model in self.classes_:
            w = self.weights[model]
            functional_form = self.functional_form
            ker = functional_form(K_list,w)
            clf = self.clfs[model]
            predicts.update({model : clf.decision_function(ker)})

        #voting
        scoring = np.zeros( (nn,self.n_classes) )
        for col,model in enumerate(self.classes_):
            scoring[:,col] = predicts[model]
        y_pred = np.array([np.argmax(sc) for sc in scoring])
        return y_pred





        

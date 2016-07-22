

from sklearn.cross_validation import train_test_split,ShuffleSplit
from sklearn.metrics import roc_auc_score,accuracy_score
import numpy as np
from cvxopt import matrix,solvers,spdiag

'''
   -------------------------------
   --- more complex heuristics ---
   -------------------------------
'''


class Heuristic():
    def evaluate(self,K,Y):
        return -1


'''
HScoring is an heuristic that assign to each kernel a value that depends on a metric used to evaluate the performance
'''
class HScoring(Heuristic):
    
    '''
    estimator : estimator/classifier used to evaluate the single kernel performance
    score : the score used to evaluate the performance, can be a callable(Y_true,Y_pred) or a string
    cv : cross-validation used in evaluation,if None then holdout method is used
    '''
    def __init__(self, estimator, score='accuracy', cv=None):
        estimator.kernel = 'precomputed'
        self.estimator = estimator
        self.score = score
        self.cv = cv


    def __score_definition__()
        if hasattr(self.score,'__call__'):
            return self.score
        elif self.score in ['auc','roc_auc']:
            return roc_auc_score
        elif self.score == 'accuracy' or True:
            return accuracy_score
    

    def evaluate(self,K,Y):
        n = len(K)
        f_score = __score_definition__()
        cv = self.cv
        if cv == None:
            cv = ShuffleSplit(n, n_iter=1, test_size=.25)

        score = []
        for train,test in cv:
            clf = self.estimator.fit(K[train][:,train],Y[train])
            y_pred = clf.predict(K[test][:,train])
            score.append(f_score(Y[test],y_pred))
        
        return np.mean(score)

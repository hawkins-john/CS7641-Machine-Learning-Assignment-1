
# -*- coding: utf-8 -*-
"""
Author: John Hawkins
Spring 2019
Code modified from online resource
Original author: JTay
Link: https://github.com/JonathanTay/CS-7641-assignment-1
"""

import numpy as np
import sklearn.model_selection as ms
import pandas as pd
from helpers import  basicResults,makeTimingCurve,iterationLC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import euclidean_distances
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel


class primalSVM_RBF(BaseEstimator, ClassifierMixin):
    '''http://scikit-learn.org/stable/developers/contributing.html'''

    def __init__(self, alpha=1e-9,gamma_frac=0.1,n_iter=2000):
         self.alpha = alpha
         self.gamma_frac = gamma_frac
         self.n_iter = n_iter

    def fit(self, X, y):
         # Check that X and y have correct shape
         X, y = check_X_y(X, y)

         # Get the kernel matrix
         dist = euclidean_distances(X,squared=True)
         median = np.median(dist)
         del dist
         gamma = median
         gamma *= self.gamma_frac
         self.gamma = 1/gamma
         kernels = rbf_kernel(X,None,self.gamma )

         self.X_ = X
         self.classes_ = unique_labels(y)
         self.kernels_ = kernels
         self.y_ = y
         self.clf = SGDClassifier(loss='hinge',penalty='l2',alpha=self.alpha,
                                  l1_ratio=0,fit_intercept=True,verbose=False,
                                  average=False,learning_rate='optimal',
                                  class_weight='balanced',n_iter=self.n_iter,
                                  random_state=55)
         self.clf.fit(self.kernels_,self.y_)

         # Return the classifier
         return self

    def predict(self, X):
         # Check is fit had been called
         check_is_fitted(self, ['X_', 'y_','clf','kernels_'])
         # Input validation
         X = check_array(X)
         new_kernels = rbf_kernel(X,self.X_,self.gamma )
         pred = self.clf.predict(new_kernels)
         return pred




# Load data
adult = pd.read_pickle('adult.pkl')
adultX = adult.drop('income',1).copy().values
adultY = adult['income'].copy().values

mushrooms = pd.read_pickle('mushrooms.pkl')
mushroomsX = mushrooms.drop('class',1).copy().values
mushroomsY = mushrooms['class'].copy().values

redwine = pd.read_pickle('winequality-red.pkl')
redwineX = redwine.drop('quality',1).copy().values
redwineY = redwine['quality'].copy().values


# Split data 70/30 between train and test in a stratified manner
adult_trgX, adult_tstX, adult_trgY, adult_tstY = ms.train_test_split(adultX, adultY, test_size=0.05, train_size=0.1666, random_state=0,stratify=adultY)
mushrooms_trgX, mushrooms_tstX, mushrooms_trgY, mushrooms_tstY = ms.train_test_split(mushroomsX, mushroomsY, test_size=0.3, random_state=0,stratify=mushroomsY)
redwine_trgX, redwine_tstX, redwine_trgY, redwine_tstY = ms.train_test_split(redwineX, redwineY, test_size=0.3, random_state=0,stratify=redwineY)


N_adult = adult_trgX.shape[0]
N_mushrooms = mushrooms_trgX.shape[0]
N_redwine = redwine_trgX.shape[0]
#N_madelon = madelon_trgX.shape[0]

alphas = [10**-x for x in np.arange(1,9.01,1/2)]


#Linear SVM
pipeA = Pipeline([('Scale',StandardScaler()),
                ('SVM',SGDClassifier(loss='hinge',l1_ratio=0,penalty='l2',class_weight='balanced',random_state=55))])
pipeR = Pipeline([('Scale',StandardScaler()),
                ('SVM',SGDClassifier(loss='hinge',l1_ratio=0,penalty='l2',class_weight='balanced',random_state=55))])

params_adult = {'SVM__alpha':alphas,'SVM__n_iter':[int((1e6/N_adult)/.8)+1]}
params_redwine = {'SVM__alpha':alphas,'SVM__n_iter':[int((1e6/N_redwine)/.8)+1]}

adult_clf = basicResults(pipeA,adult_trgX,adult_trgY,adult_tstX,adult_tstY,params_adult,'SVM_Lin','adult')
redwine_clf = basicResults(pipeR,redwine_trgX,redwine_trgY,redwine_tstX,redwine_tstY,params_redwine,'SVM_Lin','redwine')

#adult_final_params ={'SVM__alpha': 0.001, 'SVM__n_iter': 54.75}
adult_final_params =adult_clf.best_params_
adult_OF_params ={'SVM__n_iter': 55, 'SVM__alpha': 1e-16}
redwine_final_params =redwine_clf.best_params_
redwine_OF_params ={'SVM__n_iter': 55, 'SVM__alpha': 1e-16}


pipeA.set_params(**adult_final_params)
makeTimingCurve(adultX,adultY,pipeA,'SVM_Lin','adult')
pipeR.set_params(**redwine_final_params)
makeTimingCurve(redwineX,redwineY,pipeR,'SVM_Lin','redwine')

pipeA.set_params(**adult_final_params)
iterationLC(pipeA,adult_trgX,adult_trgY,adult_tstX,adult_tstY,{'SVM__n_iter':np.arange(1,75,3)},'SVM_Lin','adult')
pipeR.set_params(**redwine_final_params)
iterationLC(pipeR,redwine_trgX,redwine_trgY,redwine_tstX,redwine_tstY,{'SVM__n_iter':np.arange(1,75,3)},'SVM_Lin','redwine')

pipeA.set_params(**adult_OF_params)
iterationLC(pipeA,adult_trgX,adult_trgY,adult_tstX,adult_tstY,{'SVM__n_iter':np.arange(1,200,5)},'SVM_LinOF','adult')
pipeR.set_params(**redwine_OF_params)
iterationLC(pipeR,redwine_trgX,redwine_trgY,redwine_tstX,redwine_tstY,{'SVM__n_iter':np.arange(1,200,5)},'SVM_LinOF','redwine')






# #RBF SVM
# gamma_fracsA = np.arange(0.2,2.1,0.2)
# gamma_fracsM = np.arange(0.2,2.1,0.2)
# gamma_fracsR = np.arange(0.2,2.1,0.2)
# #gamma_fracsM = np.arange(0.05,1.01,0.1)
#
# #
# #pipeM = Pipeline([('Scale',StandardScaler()),
# #                 ('Cull1',SelectFromModel(RandomForestClassifier(random_state=1),threshold='median')),
# #                 ('Cull2',SelectFromModel(RandomForestClassifier(random_state=2),threshold='median')),
# #                 ('Cull3',SelectFromModel(RandomForestClassifier(random_state=3),threshold='median')),
# #                 ('Cull4',SelectFromModel(RandomForestClassifier(random_state=4),threshold='median')),
# #                 ('SVM',primalSVM_RBF())])
#
# # Build pipeline for feature scaling and learner
# pipeA = Pipeline([('Scale',StandardScaler()),
#                  ('SVM',primalSVM_RBF())])
# pipeM = Pipeline([('Scale',StandardScaler()),
#                  ('SVM',primalSVM_RBF())])
# pipeR = Pipeline([('Scale',StandardScaler()),
#                  ('SVM',primalSVM_RBF())])
#
#
# # Define parameters for grid search cross validation
# params_adult = {'SVM__alpha':alphas,'SVM__n_iter':[int((1e6/N_adult)/.8)+1],'SVM__gamma_frac':gamma_fracsA}
# params_mushrooms = {'SVM__alpha':alphas,'SVM__n_iter':[int((1e6/N_mushrooms)/.8)+1],'SVM__gamma_frac':gamma_fracsM}
# params_redwine = {'SVM__alpha':alphas,'SVM__n_iter':[int((1e6/N_redwine)/.8)+1],'SVM__gamma_frac':gamma_fracsR}
#
#
# # Perform grid search cross validation over the hyperparameter grid
# adult_clf = basicResults(pipeA,adult_trgX,adult_trgY,adult_tstX,adult_tstY,params_adult,'SVM_RBF','adult')
# mushrooms_clf = basicResults(pipeM,mushrooms_trgX,mushrooms_trgY,mushrooms_tstX,mushrooms_tstY,params_mushrooms,'SVM_RBF','mushrooms')
# redwine_clf = basicResults(pipeR,redwine_trgX,redwine_trgY,redwine_tstX,redwine_tstY,params_redwine,'SVM_RBF','redwine')
#
#
# # Save hyperparameters that grid search cross validation has identified as optimal
# #madelon_final_params = madelon_clf.best_params_
# #madelon_OF_params = madelon_final_params.copy()
# #madelon_OF_params['SVM__alpha'] = 1e-16
# adult_final_params =adult_clf.best_params_
# adult_OF_params = adult_final_params.copy()
# adult_OF_params['SVM__alpha'] = 1e-16
# mushrooms_final_params =mushrooms_clf.best_params_
# mushrooms_OF_params = mushrooms_final_params.copy()
# mushrooms_OF_params['SVM__alpha'] = 1e-16
# redwine_final_params =redwine_clf.best_params_
# redwine_OF_params = redwine_final_params.copy()
# redwine_OF_params['SVM__alpha'] = 1e-16
#
#
# # Feed learning algorithm optimal hyperparameters and output train/test timing curves over various train/test split ratios
# #pipeM.set_params(**madelon_final_params)
# #makeTimingCurve(madelonX,madelonY,pipeM,'SVM_RBF','madelon')
# pipeA.set_params(**adult_final_params)
# makeTimingCurve(adultX,adultY,pipeA,'SVM_RBF','adult')
# pipeM.set_params(**mushrooms_final_params)
# makeTimingCurve(mushroomsX,mushroomsY,pipeM,'SVM_RBF','mushrooms')
# pipeR.set_params(**redwine_final_params)
# makeTimingCurve(redwineX,redwineY,pipeR,'SVM_RBF','redwine')
#
#
#
# #pipeM.set_params(**madelon_final_params)
# #iterationLC(pipeM,madelon_trgX,madelon_trgY,madelon_tstX,madelon_tstY,{'SVM__n_iter':[2**x for x in range(12)]},'SVM_RBF','madelon')
# pipeA.set_params(**adult_final_params)
# iterationLC(pipeA,adult_trgX,adult_trgY,adult_tstX,adult_tstY,{'SVM__n_iter':np.arange(1,75,3)},'SVM_RBF','adult')
# pipeM.set_params(**mushrooms_final_params)
# iterationLC(pipeM,mushrooms_trgX,mushrooms_trgY,mushrooms_tstX,mushrooms_tstY,{'SVM__n_iter':np.arange(1,75,3)},'SVM_RBF','mushrooms')
# pipeR.set_params(**redwine_final_params)
# iterationLC(pipeR,redwine_trgX,redwine_trgY,redwine_tstX,redwine_tstY,{'SVM__n_iter':np.arange(1,75,3)},'SVM_RBF','redwine')
#
#
# #pipeM.set_params(**madelon_OF_params)
# #iterationLC(pipeM,madelon_trgX,madelon_trgY,madelon_tstX,madelon_tstY,{'SVM__n_iter':np.arange(100,2600,100)},'SVM_RBF_OF','madelon')
# pipeA.set_params(**adult_OF_params)
# iterationLC(pipeA,adult_trgX,adult_trgY,adult_tstX,adult_tstY,{'SVM__n_iter':np.arange(1,75,3)},'SVM_RBF_OF','adult')
# pipeM.set_params(**mushrooms_OF_params)
# iterationLC(pipeM,mushrooms_trgX,mushrooms_trgY,mushrooms_tstX,mushrooms_tstY,{'SVM__n_iter':np.arange(1,75,3)},'SVM_RBF_OF','mushrooms')
# pipeR.set_params(**redwine_OF_params)
# iterationLC(pipeR,redwine_trgX,redwine_trgY,redwine_tstX,redwine_tstY,{'SVM__n_iter':np.arange(1,75,3)},'SVM_RBF_OF','redwine')

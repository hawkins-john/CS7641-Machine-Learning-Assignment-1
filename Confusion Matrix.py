# -*- coding: utf-8 -*-
"""
Author: John Hawkins
Spring 2019
"""

import numpy as np
import sklearn.model_selection as ms
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier as knnC
import pandas as pd
from helpers import  basicResults,dtclf_pruned, makeTimingCurve
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
from sklearn.metrics import confusion_matrix


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
redwine_trgX, redwine_tstX, redwine_trgY, redwine_tstY = ms.train_test_split(redwineX, redwineY, test_size=0.3, random_state=0,stratify=redwineY)

# DT
pipeA = Pipeline([('Scale',StandardScaler()),
                 ('DT',dtclf_pruned(random_state=55))])

pipeR = Pipeline([('Scale',StandardScaler()),
                 ('DT',dtclf_pruned(random_state=55))])

adult_final_params = {'DT__alpha': 0.0031622776601683794, 'DT__class_weight': 'balanced', 'DT__criterion': 'entropy'}
redwine_final_params = {'DT__alpha': -0.0316227766016838, 'DT__class_weight': 'balanced', 'DT__criterion': 'entropy'}

pipeA.set_params(**adult_final_params)
pipeR.set_params(**redwine_final_params)

pipeA.fit(adult_trgX, adult_trgY)
pipeR.fit(redwine_trgX, redwine_trgY)
predA = pipeA.predict(adult_tstX)
predR = pipeR.predict(redwine_tstX)
confA = confusion_matrix(adult_tstY, predA, labels = [1, 0])
confR = confusion_matrix(redwine_tstY, predR, labels = [0, 1])
print(confA)
print(confR)



# Neural Network
pipeA = Pipeline([('Scale',StandardScaler()),
                 ('MLP',MLPClassifier(max_iter=2000,early_stopping=True,random_state=55))])
pipeR = Pipeline([('Scale',StandardScaler()),
                 ('MLP',MLPClassifier(max_iter=2000,early_stopping=True,random_state=55))])

adult_final_params ={'MLP__hidden_layer_sizes': (28, 28, 28), 'MLP__activation': 'logistic', 'MLP__alpha': 0.0031622776601683794}
redwine_final_params ={'MLP__hidden_layer_sizes': (28, 28, 28), 'MLP__activation': 'logistic', 'MLP__alpha': 0.0031622776601683794}

pipeA.set_params(**adult_final_params)
pipeR.set_params(**redwine_final_params)

pipeA.fit(adult_trgX, adult_trgY)
pipeR.fit(redwine_trgX, redwine_trgY)
predA = pipeA.predict(adult_tstX)
predR = pipeR.predict(redwine_tstX)
confA = confusion_matrix(adult_tstY, predA, labels = [1, 0])
confR = confusion_matrix(redwine_tstY, predR, labels = [0, 1])
print(confA)
print(confR)



# Boosting
adult_base = dtclf_pruned(criterion='entropy',class_weight='balanced',random_state=55)
redwine_base = dtclf_pruned(criterion='entropy',class_weight='balanced',random_state=55)
adult_booster = AdaBoostClassifier(algorithm='SAMME',learning_rate=1,base_estimator=adult_base,random_state=55)
redwine_booster = AdaBoostClassifier(algorithm='SAMME',learning_rate=1,base_estimator=redwine_base,random_state=55)
pipeA = Pipeline([('Scale',StandardScaler()),
                 ('Boost',adult_booster)])
pipeR = Pipeline([('Scale',StandardScaler()),
                 ('Boost',redwine_booster)])

adult_final_params = {'Boost__base_estimator__alpha':0.01, 'Boost__n_estimators':10}
redwine_final_params = {'Boost__base_estimator__alpha':0, 'Boost__n_estimators':10}

pipeA.set_params(**adult_final_params)
pipeR.set_params(**redwine_final_params)

pipeA.fit(adult_trgX, adult_trgY)
pipeR.fit(redwine_trgX, redwine_trgY)
predA = pipeA.predict(adult_tstX)
predR = pipeR.predict(redwine_tstX)
confA = confusion_matrix(adult_tstY, predA, labels = [1, 0])
confR = confusion_matrix(redwine_tstY, predR, labels = [0, 1])
print(confA)
print(confR)



# Linear SVM
pipeA = Pipeline([('Scale',StandardScaler()),
                ('SVM',SGDClassifier(loss='hinge',l1_ratio=0,penalty='l2',class_weight='balanced',random_state=55))])
pipeR = Pipeline([('Scale',StandardScaler()),
                ('SVM',SGDClassifier(loss='hinge',l1_ratio=0,penalty='l2',class_weight='balanced',random_state=55))])

adult_final_params={'SVM__alpha': 0.001, 'SVM__n_iter': 231}
redwine_final_params={'SVM__alpha': 0.01, 'SVM__n_iter': 1118}

pipeA.set_params(**adult_final_params)
pipeR.set_params(**redwine_final_params)

pipeA.fit(adult_trgX, adult_trgY)
pipeR.fit(redwine_trgX, redwine_trgY)
predA = pipeA.predict(adult_tstX)
predR = pipeR.predict(redwine_tstX)
confA = confusion_matrix(adult_tstY, predA, labels = [1, 0])
confR = confusion_matrix(redwine_tstY, predR, labels = [0, 1])
print(confA)
print(confR)



# RBF SVM
pipeA = Pipeline([('Scale',StandardScaler()),
                 ('SVM',primalSVM_RBF())])
pipeR = Pipeline([('Scale',StandardScaler()),
                 ('SVM',primalSVM_RBF())])


adult_final_params={'SVM__alpha': 0.0031622776601683794,'SVM__n_iter':231,'SVM__gamma_frac':1.4}
redwine_final_params={'SVM__alpha':0.001,'SVM__n_iter':1118,'SVM__gamma_frac':0.2}

pipeA.set_params(**adult_final_params)
pipeR.set_params(**redwine_final_params)

pipeA.fit(adult_trgX, adult_trgY)
pipeR.fit(redwine_trgX, redwine_trgY)
predA = pipeA.predict(adult_tstX)
predR = pipeR.predict(redwine_tstX)
confA = confusion_matrix(adult_tstY, predA, labels = [1, 0])
confR = confusion_matrix(redwine_tstY, predR, labels = [0, 1])
print(confA)
print(confR)



# kNN
pipeA = Pipeline([('Scale',StandardScaler()),
                 ('KNN',knnC())])
pipeR = Pipeline([('Scale',StandardScaler()),
                 ('KNN',knnC())])

adult_final_params={'KNN__n_neighbors': 4, 'KNN__p': 1, 'KNN__weights': 'uniform'}
redwine_final_params={'KNN__n_neighbors': 34, 'KNN__p': 1, 'KNN__weights': 'distance'}

pipeA.set_params(**adult_final_params)
pipeR.set_params(**redwine_final_params)

pipeA.fit(adult_trgX, adult_trgY)
pipeR.fit(redwine_trgX, redwine_trgY)
predA = pipeA.predict(adult_tstX)
predR = pipeR.predict(redwine_tstX)
confA = confusion_matrix(adult_tstY, predA, labels = [1, 0])
confR = confusion_matrix(redwine_tstY, predR, labels = [0, 1])
print(confA)
print(confR)

# -*- coding: utf-8 -*-
"""
Author: John Hawkins
Spring 2019
Code modified from online resource
Original author: JTay
Link: https://github.com/JonathanTay/CS-7641-assignment-1
"""


import sklearn.model_selection as ms
from sklearn.ensemble import AdaBoostClassifier
from helpers import dtclf_pruned
import pandas as pd
from helpers import  basicResults,makeTimingCurve,iterationLC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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


# Search for good alphas
alphas = [-1,-1e-3,-(1e-3)*10**-0.5, -1e-2, -(1e-2)*10**-0.5,-1e-1,-(1e-1)*10**-0.5, 0, (1e-1)*10**-0.5,1e-1,(1e-2)*10**-0.5,1e-2,(1e-3)*10**-0.5,1e-3]


#madelon_base = dtclf_pruned(criterion='gini',class_weight='balanced',random_state=55)
adult_base = dtclf_pruned(criterion='entropy',class_weight='balanced',random_state=55)
mushrooms_base = dtclf_pruned(criterion='entropy',class_weight='balanced',random_state=55)
redwine_base = dtclf_pruned(criterion='entropy',class_weight='balanced',random_state=55)
OF_base = dtclf_pruned(criterion='gini',class_weight='balanced',random_state=55)


# Define parameters for grid search cross validation
#paramsA= {'Boost__n_estimators':[1,2,5,10,20,30,40,50],'Boost__learning_rate':[(2**x)/100 for x in range(8)]+[1]}
paramsA= {'Boost__n_estimators':[1,2,5,10,20,30,45,60,80,100],
          'Boost__base_estimator__alpha':alphas}
paramsM= {'Boost__n_estimators':[1,2,5,10,20,30,45,60,80,100],
          'Boost__base_estimator__alpha':alphas}
paramsR= {'Boost__n_estimators':[1,2,5,10,20,30,45,60,80,100],
          'Boost__base_estimator__alpha':alphas}
#paramsM = {'Boost__n_estimators':[1,2,5,10,20,30,40,50,60,70,80,90,100],
#           'Boost__learning_rate':[(2**x)/100 for x in range(8)]+[1]}
#paramsM = {'Boost__n_estimators':[1,2,5,10,20,30,45,60,80,100],
#           'Boost__base_estimator__alpha':alphas}


# Build AdaBoost classifiers
#madelon_booster = AdaBoostClassifier(algorithm='SAMME',learning_rate=1,base_estimator=madelon_base,random_state=55)
adult_booster = AdaBoostClassifier(algorithm='SAMME',learning_rate=1,base_estimator=adult_base,random_state=55)
mushrooms_booster = AdaBoostClassifier(algorithm='SAMME',learning_rate=1,base_estimator=mushrooms_base,random_state=55)
redwine_booster = AdaBoostClassifier(algorithm='SAMME',learning_rate=1,base_estimator=redwine_base,random_state=55)
OF_booster = AdaBoostClassifier(algorithm='SAMME',learning_rate=1,base_estimator=OF_base,random_state=55)

#pipeM = Pipeline([('Scale',StandardScaler()),
#                 ('Cull1',SelectFromModel(RandomForestClassifier(random_state=1),threshold='median')),
#                 ('Cull2',SelectFromModel(RandomForestClassifier(random_state=2),threshold='median')),
#                 ('Cull3',SelectFromModel(RandomForestClassifier(random_state=3),threshold='median')),
#                 ('Cull4',SelectFromModel(RandomForestClassifier(random_state=4),threshold='median')),
#                 ('Boost',madelon_booster)])

# Build pipeline for feature scaling and learner
pipeA = Pipeline([('Scale',StandardScaler()),
                 ('Boost',adult_booster)])
pipeM = Pipeline([('Scale',StandardScaler()),
                 ('Boost',mushrooms_booster)])
pipeR = Pipeline([('Scale',StandardScaler()),
                 ('Boost',redwine_booster)])

# Perform grid search cross validation over the hyperparameter grid
#madelon_clf = basicResults(pipeM,madelon_trgX,madelon_trgY,madelon_tstX,madelon_tstY,paramsM,'Boost','madelon')
adult_clf = basicResults(pipeA,adult_trgX,adult_trgY,adult_tstX,adult_tstY,paramsA,'Boost','adult')
mushrooms_clf = basicResults(pipeM,mushrooms_trgX,mushrooms_trgY,mushrooms_tstX,mushrooms_tstY,paramsM,'Boost','mushrooms')
redwine_clf = basicResults(pipeR,redwine_trgX,redwine_trgY,redwine_tstX,redwine_tstY,paramsR,'Boost','redwine')

#
#madelon_final_params = {'n_estimators': 20, 'learning_rate': 0.02}
#adult_final_params = {'n_estimators': 10, 'learning_rate': 1}
#OF_params = {'learning_rate':1}


# Save hyperparameters that grid search cross validation has identified as optimal
#madelon_final_params = madelon_clf.best_params_
adult_final_params = adult_clf.best_params_
mushrooms_final_params = mushrooms_clf.best_params_
redwine_final_params = redwine_clf.best_params_
OF_params = {'Boost__base_estimator__alpha':-1, 'Boost__n_estimators':50}


# Feed learning algorithm optimal hyperparameters and output train/test timing curves over various train/test split ratios
#pipeM.set_params(**madelon_final_params)
#makeTimingCurve(madelonX,madelonY,pipeM,'Boost','madelon')
pipeA.set_params(**adult_final_params)
makeTimingCurve(adultX,adultY,pipeA,'Boost','adult')
pipeM.set_params(**mushrooms_final_params)
makeTimingCurve(mushroomsX,mushroomsY,pipeM,'Boost','mushrooms')
pipeR.set_params(**redwine_final_params)
makeTimingCurve(redwineX,redwineY,pipeR,'Boost','redwine')


#
#pipeM.set_params(**madelon_final_params)
#iterationLC(pipeM,madelon_trgX,madelon_trgY,madelon_tstX,madelon_tstY,{'Boost__n_estimators':[1,2,5,10,20,30,40,50,60,70,80,90,100]},'Boost','madelon')
pipeA.set_params(**adult_final_params)
iterationLC(pipeA,adult_trgX,adult_trgY,adult_tstX,adult_tstY,{'Boost__n_estimators':[1,2,5,10,20,30,40,50]},'Boost','adult')
pipeM.set_params(**mushrooms_final_params)
iterationLC(pipeM,mushrooms_trgX,mushrooms_trgY,mushrooms_tstX,mushrooms_tstY,{'Boost__n_estimators':[1,2,5,10,20,30,40,50]},'Boost','mushrooms')
pipeR.set_params(**redwine_final_params)
iterationLC(pipeR,redwine_trgX,redwine_trgY,redwine_tstX,redwine_tstY,{'Boost__n_estimators':[1,2,5,10,20,30,40,50]},'Boost','redwine')
#pipeM.set_params(**OF_params)
#iterationLC(pipeM,madelon_trgX,madelon_trgY,madelon_tstX,madelon_tstY,{'Boost__n_estimators':[1,2,5,10,20,30,40,50,60,70,80,90,100]},'Boost_OF','madelon')
pipeA.set_params(**OF_params)
iterationLC(pipeA,adult_trgX,adult_trgY,adult_tstX,adult_tstY,{'Boost__n_estimators':[1,2,5,10,20,30,40,50]},'Boost_OF','adult')
pipeM.set_params(**OF_params)
iterationLC(pipeM,mushrooms_trgX,mushrooms_trgY,mushrooms_tstX,mushrooms_tstY,{'Boost__n_estimators':[1,2,5,10,20,30,40,50]},'Boost_OF','mushrooms')
pipeR.set_params(**OF_params)
iterationLC(pipeR,redwine_trgX,redwine_trgY,redwine_tstX,redwine_tstY,{'Boost__n_estimators':[1,2,5,10,20,30,40,50]},'Boost_OF','redwine')

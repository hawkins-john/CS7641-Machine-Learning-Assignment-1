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
from sklearn.neighbors import KNeighborsClassifier as knnC
import pandas as pd
from helpers import  basicResults,makeTimingCurve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel


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


#pipeM = Pipeline([('Scale',StandardScaler()),
#                 ('Cull1',SelectFromModel(RandomForestClassifier(),threshold='median')),
#                 ('Cull2',SelectFromModel(RandomForestClassifier(),threshold='median')),
#                 ('Cull3',SelectFromModel(RandomForestClassifier(),threshold='median')),
#                 ('Cull4',SelectFromModel(RandomForestClassifier(),threshold='median')),
#                 ('KNN',knnC())])

# Build pipeline for feature scaling and learner
pipeA = Pipeline([('Scale',StandardScaler()),
                 ('KNN',knnC())])

pipeM = Pipeline([('Scale',StandardScaler()),
                 ('KNN',knnC())])

pipeR = Pipeline([('Scale',StandardScaler()),
                 ('KNN',knnC())])


# Define hyperparameters for grid search cross validation
params_adult= {'KNN__metric':['manhattan','euclidean','chebyshev'],'KNN__n_neighbors':np.arange(1,51,3),'KNN__weights':['uniform','distance']}
params_mushrooms= {'KNN__metric':['manhattan','euclidean','chebyshev'],'KNN__n_neighbors':np.arange(1,51,3),'KNN__weights':['uniform','distance']}
params_redwine = {'KNN__metric':['manhattan','euclidean','chebyshev'],'KNN__n_neighbors':np.arange(1,51,3),'KNN__weights':['uniform','distance']}


# Perform grid search cross validation over the hyperparameter grid
adult_clf = basicResults(pipeA,adult_trgX,adult_trgY,adult_tstX,adult_tstY,params_adult,'KNN','adult')
mushrooms_clf = basicResults(pipeM,mushrooms_trgX,mushrooms_trgY,mushrooms_tstX,mushrooms_tstY,params_mushrooms,'KNN','mushrooms')
redwine_clf = basicResults(pipeR,redwine_trgX,redwine_trgY,redwine_tstX,redwine_tstY,params_redwine,'KNN','redwine')


# Save hyperparameters that grid search cross validation has identified as optimal
#madelon_final_params={'KNN__n_neighbors': 43, 'KNN__weights': 'uniform', 'KNN__p': 1}
#adult_final_params={'KNN__n_neighbors': 142, 'KNN__p': 1, 'KNN__weights': 'uniform'}
adult_final_params=adult_clf.best_params_
mushrooms_final_params=mushrooms_clf.best_params_
redwine_final_params=redwine_clf.best_params_


# Feed learning algorithm optimal hyperparameters and output train/test timing curves over various train/test split ratios
pipeA.set_params(**adult_final_params)
makeTimingCurve(adultX,adultY,pipeA,'KNN','adult')
pipeM.set_params(**mushrooms_final_params)
makeTimingCurve(mushroomsX,mushroomsY,pipeM,'KNN','mushrooms')
pipeR.set_params(**redwine_final_params)
makeTimingCurve(redwineX,redwineY,pipeR,'KNN','redwine')

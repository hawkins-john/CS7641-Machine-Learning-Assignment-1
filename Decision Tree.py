# -*- coding: utf-8 -*-
"""
Author: John Hawkins
Spring 2019
Code modified from online resource
Original author: JTay
Link: https://github.com/JonathanTay/CS-7641-assignment-1
"""

import sklearn.model_selection as ms
import pandas as pd
from helpers import basicResults,dtclf_pruned,makeTimingCurve
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def DTpruningVSnodes(clf,alphas,trgX,trgY,dataset):
    '''Dump table of pruning alpha vs. # of internal nodes'''
    out = {}
    for a in alphas:
        clf.set_params(**{'DT__alpha':a})
        clf.fit(trgX,trgY)
        out[a]=clf.steps[-1][-1].numNodes()
        print(dataset,a)
    out = pd.Series(out)
    out.index.name='alpha'
    out.name = 'Number of Internal Nodes'
    out.to_csv('./output/DT_{}_nodecounts.csv'.format(dataset))

    return


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
#alphas=[0]

#pipeM = Pipeline([('Scale',StandardScaler()),
#                 ('Cull1',SelectFromModel(RandomForestClassifier(random_state=1),threshold='median')),
#                 ('Cull2',SelectFromModel(RandomForestClassifier(random_state=2),threshold='median')),
#                 ('Cull3',SelectFromModel(RandomForestClassifier(random_state=3),threshold='median')),
#                 ('Cull4',SelectFromModel(RandomForestClassifier(random_state=4),threshold='median')),
#                 ('DT',dtclf_pruned(random_state=55))])


# Build pipeline for feature scaling and learner
pipeA = Pipeline([('Scale',StandardScaler()),
                 ('DT',dtclf_pruned(random_state=55))])
pipeM = Pipeline([('Scale',StandardScaler()),
                 ('DT',dtclf_pruned(random_state=55))])
pipeR = Pipeline([('Scale',StandardScaler()),
                 ('DT',dtclf_pruned(random_state=55))])


# Define hyperparameters for grid search cross validation
params = {'DT__criterion':['gini','entropy'],'DT__alpha':alphas,'DT__class_weight':['balanced']}


# Perform grid search cross validation over the hyperparameter grid
adult_clf = basicResults(pipeA,adult_trgX,adult_trgY,adult_tstX,adult_tstY,params,'DT','adult')
mushrooms_clf = basicResults(pipeM,mushrooms_trgX,mushrooms_trgY,mushrooms_tstX,mushrooms_tstY,params,'DT','mushrooms')
redwine_clf = basicResults(pipeR,redwine_trgX,redwine_trgY,redwine_tstX,redwine_tstY,params,'DT','redwine')


# Save hyperparameters that grid search cross validation has identified as optimal
#madelon_final_params = {'DT__alpha': -0.00031622776601683794, 'DT__class_weight': 'balanced', 'DT__criterion': 'entropy'}
#adult_final_params = {'class_weight': 'balanced', 'alpha': 0.0031622776601683794, 'criterion': 'entropy'}
adult_final_params = adult_clf.best_params_
mushrooms_final_params = mushrooms_clf.best_params_
redwine_final_params = redwine_clf.best_params_


# Feed learning algorithm optimal hyperparameters and output train/test timing curves over various train/test split ratios
pipeA.set_params(**adult_final_params)
makeTimingCurve(adultX,adultY,pipeA,'DT','adult')
pipeM.set_params(**mushrooms_final_params)
makeTimingCurve(mushroomsX,mushroomsY,pipeM,'DT','mushrooms')
pipeR.set_params(**redwine_final_params)
makeTimingCurve(redwineX,redwineY,pipeR,'DT','redwine')


# Output number of tree nodes versus pruning alpha
DTpruningVSnodes(pipeA,alphas,adult_trgX,adult_trgY,'adult')
DTpruningVSnodes(pipeM,alphas,mushrooms_trgX,mushrooms_trgY,'mushrooms')
DTpruningVSnodes(pipeR,alphas,redwine_trgX,redwine_trgY,'redwine')

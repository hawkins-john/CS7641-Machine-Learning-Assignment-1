# -*- coding: utf-8 -*-
"""
Author: John Hawkins
Spring 2019
Code modified from online resource
Original author: JTay
Link: https://github.com/JonathanTay/CS-7641-assignment-1
"""

import numpy as np
from sklearn.neural_network import MLPClassifier
import sklearn.model_selection as ms
import pandas as pd
from helpers import  basicResults,makeTimingCurve,iterationLC
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
#                 ('Cull1',SelectFromModel(RandomForestClassifier(random_state=1),threshold='median')),
#                 ('Cull2',SelectFromModel(RandomForestClassifier(random_state=2),threshold='median')),
#                 ('Cull3',SelectFromModel(RandomForestClassifier(random_state=3),threshold='median')),
#                 ('Cull4',SelectFromModel(RandomForestClassifier(random_state=4),threshold='median')),
#                 ('MLP',MLPClassifier(max_iter=2000,early_stopping=True,random_state=55))])

# Build pipeline for feature scaling and learner
pipeA = Pipeline([('Scale',StandardScaler()),
                 ('MLP',MLPClassifier(max_iter=2000,early_stopping=True,random_state=55))])
pipeM = Pipeline([('Scale',StandardScaler()),
                 ('MLP',MLPClassifier(max_iter=2000,early_stopping=True,random_state=55))])
pipeR = Pipeline([('Scale',StandardScaler()),
                 ('MLP',MLPClassifier(max_iter=2000,early_stopping=True,random_state=55))])


# Define parameters for grid search cross validation
d = adultX.shape[1]
hiddens_adult = [(h,)*l for l in [1,2,3] for h in [d,d//2,d*2]]
d = mushroomsX.shape[1]
hiddens_mushrooms = [(h,)*l for l in [1,2,3] for h in [d,d//2,d*2]]
d = redwineX.shape[1]
hiddens_redwine = [(h,)*l for l in [1,2,3] for h in [d,d//2,d*2]]
alphas = [10**-x for x in np.arange(-1,5.01,1/2)]
#alphasM = [10**-x for x in np.arange(-1,9.01,1/2)]
#d = madelonX.shape[1]
#d = d//(2**4)
#hiddens_madelon = [(h,)*l for l in [1,2,3] for h in [d,d//2,d*2]]
params_adult = {'MLP__activation':['relu','logistic'],'MLP__alpha':alphas,'MLP__hidden_layer_sizes':hiddens_adult}
params_mushrooms = {'MLP__activation':['relu','logistic'],'MLP__alpha':alphas,'MLP__hidden_layer_sizes':hiddens_mushrooms}
params_redwine = {'MLP__activation':['relu','logistic'],'MLP__alpha':alphas,'MLP__hidden_layer_sizes':hiddens_redwine}


# Perform grid search cross validation over the hyperparameter grid
adult_clf = basicResults(pipeA,adult_trgX,adult_trgY,adult_tstX,adult_tstY,params_adult,'ANN','adult')
mushrooms_clf = basicResults(pipeM,mushrooms_trgX,mushrooms_trgY,mushrooms_tstX,mushrooms_tstY,params_mushrooms,'ANN','mushrooms')
redwine_clf = basicResults(pipeR,redwine_trgX,redwine_trgY,redwine_tstX,redwine_tstY,params_redwine,'ANN','redwine')


#madelon_final_params = {'MLP__hidden_layer_sizes': (500,), 'MLP__activation': 'logistic', 'MLP__alpha': 10.0}
#adult_final_params ={'MLP__hidden_layer_sizes': (28, 28, 28), 'MLP__activation': 'logistic', 'MLP__alpha': 0.0031622776601683794}


# Save hyperparameters that grid search cross validation has identified as optimal
#madelon_final_params = madelon_clf.best_params_
adult_final_params =adult_clf.best_params_
adult_OF_params =adult_final_params.copy()
adult_OF_params['MLP__alpha'] = 0
mushrooms_final_params =mushrooms_clf.best_params_
mushrooms_OF_params =mushrooms_final_params.copy()
mushrooms_OF_params['MLP__alpha'] = 0
redwine_final_params =redwine_clf.best_params_
redwine_OF_params =redwine_final_params.copy()
redwine_OF_params['MLP__alpha'] = 0
#madelon_OF_params =madelon_final_params.copy()
#madelon_OF_params['MLP__alpha'] = 0

#raise


# Feed learning algorithm optimal hyperparameters and output train/test timing curves over various train/test split ratios
#pipeM.set_params(**madelon_final_params)
#pipeM.set_params(**{'MLP__early_stopping':False})
#makeTimingCurve(madelonX,madelonY,pipeM,'ANN','madelon')
pipeA.set_params(**adult_final_params)
pipeA.set_params(**{'MLP__early_stopping':False})
makeTimingCurve(adultX,adultY,pipeA,'ANN','adult')
pipeM.set_params(**mushrooms_final_params)
pipeM.set_params(**{'MLP__early_stopping':False})
makeTimingCurve(mushroomsX,mushroomsY,pipeM,'ANN','mushrooms')
pipeR.set_params(**redwine_final_params)
pipeR.set_params(**{'MLP__early_stopping':False})
makeTimingCurve(redwineX,redwineY,pipeR,'ANN','redwine')


#pipeM.set_params(**madelon_final_params)
#pipeM.set_params(**{'MLP__early_stopping':False})
#iterationLC(pipeM,madelon_trgX,madelon_trgY,madelon_tstX,madelon_tstY,{'MLP__max_iter':[2**x for x in range(12)]+[2100,2200,2300,2400,2500,2600,2700,2800,2900,3000]},'ANN','madelon')
pipeA.set_params(**adult_final_params)
pipeA.set_params(**{'MLP__early_stopping':False})
iterationLC(pipeA,adult_trgX,adult_trgY,adult_tstX,adult_tstY,{'MLP__max_iter':[2**x for x in range(12)]+[2100,2200,2300,2400,2500,2600,2700,2800,2900,3000]},'ANN','adult')
pipeM.set_params(**mushrooms_final_params)
pipeM.set_params(**{'MLP__early_stopping':False})
iterationLC(pipeM,mushrooms_trgX,mushrooms_trgY,mushrooms_tstX,mushrooms_tstY,{'MLP__max_iter':[2**x for x in range(12)]+[2100,2200,2300,2400,2500,2600,2700,2800,2900,3000]},'ANN','mushrooms')
pipeR.set_params(**redwine_final_params)
pipeR.set_params(**{'MLP__early_stopping':False})
iterationLC(pipeR,redwine_trgX,redwine_trgY,redwine_tstX,redwine_tstY,{'MLP__max_iter':[2**x for x in range(12)]+[2100,2200,2300,2400,2500,2600,2700,2800,2900,3000]},'ANN','redwine')

#pipeM.set_params(**madelon_OF_params)
#pipeM.set_params(**{'MLP__early_stopping':False})
#iterationLC(pipeM,madelon_trgX,madelon_trgY,madelon_tstX,madelon_tstY,{'MLP__max_iter':[2**x for x in range(12)]+[2100,2200,2300,2400,2500,2600,2700,2800,2900,3000]},'ANN_OF','madelon')
pipeA.set_params(**adult_OF_params)
pipeA.set_params(**{'MLP__early_stopping':False})
iterationLC(pipeA,adult_trgX,adult_trgY,adult_tstX,adult_tstY,{'MLP__max_iter':[2**x for x in range(12)]+[2100,2200,2300,2400,2500,2600,2700,2800,2900,3000]},'ANN_OF','adult')
pipeM.set_params(**mushrooms_OF_params)
pipeM.set_params(**{'MLP__early_stopping':False})
iterationLC(pipeM,mushrooms_trgX,mushrooms_trgY,mushrooms_tstX,mushrooms_tstY,{'MLP__max_iter':[2**x for x in range(12)]+[2100,2200,2300,2400,2500,2600,2700,2800,2900,3000]},'ANN_OF','mushrooms')
pipeR.set_params(**redwine_OF_params)
pipeR.set_params(**{'MLP__early_stopping':False})
iterationLC(pipeR,redwine_trgX,redwine_trgY,redwine_tstX,redwine_tstY,{'MLP__max_iter':[2**x for x in range(12)]+[2100,2200,2300,2400,2500,2600,2700,2800,2900,3000]},'ANN_OF','redwine')

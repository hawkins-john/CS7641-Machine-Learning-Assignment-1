# -*- coding: utf-8 -*-
"""
Author: John Hawkins
Spring 2019
Code modified from online resource
Original author: JTay
Link: https://github.com/JonathanTay/CS-7641-assignment-1
"""

import pandas as pd
import numpy as np

# Preprocess with adult dataset
adult = pd.read_csv('./adult.data.txt',header=None)
adult.columns = ['age','employer','fnlwt','edu','edu_num','marital','occupation','relationship','race','sex','cap_gain','cap_loss','hrs','country','income']
#print(adult.groupby('income').income.count())
# Note that cap_gain > 0 => cap_loss = 0 and vice versa. Combine variables.
print(adult.ix[adult.cap_gain>0].cap_loss.abs().max())
print(adult.ix[adult.cap_loss>0].cap_gain.abs().max())
adult['cap_gain_loss'] = adult['cap_gain']-adult['cap_loss']
adult = adult.drop(['fnlwt','edu','cap_gain','cap_loss'],1)
adult['income'] = pd.get_dummies(adult.income)
print(adult.groupby('occupation')['occupation'].count())
print(adult.groupby('country').country.count())
#http://scg.sdsu.edu/dataset-adult_r/
replacements = { 'Cambodia':' SE-Asia',
                'Canada':' British-Commonwealth',
                'China':' China',
                'Columbia':' South-America',
                'Cuba':' Other',
                'Dominican-Republic':' Latin-America',
                'Ecuador':' South-America',
                'El-Salvador':' South-America ',
                'England':' British-Commonwealth',
                'France':' Euro_1',
                'Germany':' Euro_1',
                'Greece':' Euro_2',
                'Guatemala':' Latin-America',
                'Haiti':' Latin-America',
                'Holand-Netherlands':' Euro_1',
                'Honduras':' Latin-America',
                'Hong':' China',
                'Hungary':' Euro_2',
                'India':' British-Commonwealth',
                'Iran':' Other',
                'Ireland':' British-Commonwealth',
                'Italy':' Euro_1',
                'Jamaica':' Latin-America',
                'Japan':' Other',
                'Laos':' SE-Asia',
                'Mexico':' Latin-America',
                'Nicaragua':' Latin-America',
                'Outlying-US(Guam-USVI-etc)':' Latin-America',
                'Peru':' South-America',
                'Philippines':' SE-Asia',
                'Poland':' Euro_2',
                'Portugal':' Euro_2',
                'Puerto-Rico':' Latin-America',
                'Scotland':' British-Commonwealth',
                'South':' Euro_2',
                'Taiwan':' China',
                'Thailand':' SE-Asia',
                'Trinadad&Tobago':' Latin-America',
                'United-States':' United-States',
                'Vietnam':' SE-Asia',
                'Yugoslavia':' Euro_2'}
adult['country'] = adult['country'].str.strip()
adult = adult.replace(to_replace={'country':replacements,
                                  'employer':{' Without-pay': ' Never-worked'},
                                  'relationship':{' Husband': 'Spouse',' Wife':'Spouse'}})
adult['country'] = adult['country'].str.strip()
print(adult.groupby('country').country.count())
for col in ['employer','marital','occupation','relationship','race','sex','country']:
    adult[col] = adult[col].str.strip()
#print(adult.shape)
#print(adult.columns.values)
adult = pd.get_dummies(adult)
#print(adult.shape)
#print(adult.columns.values)
print(adult.groupby('income').income.count())
adult = adult.rename(columns=lambda x: x.replace('-','_'))

#adult.to_hdf('datasets.hdf','adult',complib='blosc',complevel=9)
adult.to_pickle('adult.pkl')



# Preprocess with mushrooms dataset
mushrooms = pd.read_csv('./mushrooms.csv')
print(list(mushrooms))
#print(mushrooms.groupby('class')['class'].count())
mushrooms['class'] = pd.get_dummies(mushrooms['class'])
#print(mushrooms.groupby('class')['class'].count())

mushrooms = mushrooms.drop(['veil-type','veil-color','gill-attachment'],1)

for col in ['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-spacing',\
 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring',\
 'stalk-color-above-ring', 'stalk-color-below-ring', 'ring-number', 'ring-type',\
 'spore-print-color', 'population', 'habitat']:
    mushrooms[col] = mushrooms[col].str.strip()
mushrooms = mushrooms.rename(columns=lambda x: x.replace('-','_'))

#print(mushrooms.shape)
mushrooms = pd.get_dummies(mushrooms)
#print(mushrooms.shape)
#print(list(mushrooms['class']))
mushrooms.to_pickle('mushrooms.pkl')



# Preprocess with mushrooms dataset
redwine = pd.read_csv('./winequality-red.csv')
print(list(redwine))
print(redwine.groupby('quality')['quality'].count())

redwine.loc[redwine.quality < 5.5, 'quality'] = 0
redwine.loc[redwine['quality'] > 5.5, 'quality'] = 1
print(redwine.groupby('quality')['quality'].count())

redwine.to_pickle('winequality-red.pkl')

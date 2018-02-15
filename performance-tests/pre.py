#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 09:28:56 2018

@author: rost
"""
from pandas import *

train = read_table('data/train.csv', sep=',')

train.loc[train.loc[:,'Age'].isnull(),'Age'] = train.Age.median()

MaxPassEmbarked = train.groupby('Embarked').count()['PassengerId']

train.loc[train.loc[:,'Embarked'].isnull(),'Embarked'] = MaxPassEmbarked[MaxPassEmbarked == MaxPassEmbarked.max()].index[0]

train = train.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

label = LabelEncoder()
dicts = {}

label.fit(train.Sex.drop_duplicates())
dicts['Sex'] = list(label.classes_)
train.Sex = label.transform(train.Sex) 

label.fit(train.Embarked.drop_duplicates())
dicts['Embarked'] = list(label.classes_)
train.Embarked = label.transform(train.Embarked)


target = train.Survived
train = train.drop(['Survived'], axis=1)

# X_train, X_test, y_train, y_test = (x.values for x in train_test_split(train, target, test_size=0.2, random_state=0))

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
# pca = PCA(n_components=2).fit_transform(train)
# X_train, X_test, y_train, y_test = train_test_split(pca, target, test_size=0.2, random_state=0)

from sklearn import datasets
iris = datasets.load_iris()
iris.target[iris.target == 1] = 0
iris.target[iris.target == 2] = 1
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

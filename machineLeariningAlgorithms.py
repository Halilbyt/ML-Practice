# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 16:21:06 2021

@author: Halil İbrahim BAYAT

Genel Tekarar:
    Regression, classifcation
    Deep Learning
    Model Selectin and finding best parameters
    
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score,r2_score
from sklearn.svm import SVC,SVR
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
 

# Task First : Prediction
'''
# Loading Data and pre-processing 

data = pd.read_excel('merc.xlsx')

print('DataSets of Merc cars price')
print(data.describe())
print('Dataset colums and some rows')
print(data.head())
print('Check the data for emthy or null value')
print(data.isnull().any())
print('Corelation result of the DataBase')
print(data.corr())


x=data.drop(['price','transmission'], axis=1).values
y=data.iloc[:,1:2].values


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33)


# Linear Regression

lr=LinearRegression()
lr.fit(x_train,y_train)

y_pred=lr.predict(x_test)
results_1=pd.DataFrame(data=np.concatenate((y_test,y_pred),axis=1),columns=['Real','Predic'])

r2_1=r2_score(y_test,y_pred)
print(r2_1)

# Multi-Linear Regression


#lr=LinearRegression()
#lr.fit(x_train,y_train)

#y_pred=lr.predict(x_test)
#results=pd.DataFrame(data=np.concatenate((y_test,y_pred),axis=1),columns=['Real','Predic'])
#print(results)


# KNN

knn=KNeighborsRegressor(n_neighbors=10)
knn.fit(x_train,y_train.ravel())

y_pred=knn.predict(x_test)
r2_2=r2_score(y_test,y_pred)
print(r2_2)

# Decision Tree

dt=DecisionTreeRegressor()
dt.fit(x_train,y_train.ravel())
y_pred=dt.predict(x_test)

r2_3=r2_score(y_test,y_pred)
print(r2_3)

# Random Forest

rf=RandomForestRegressor(n_estimators=100,random_state=0)

rf.fit(x_train,y_train.ravel())
y_pred=rf.predict(x_test)

r2_4=r2_score(y_test,y_pred)
print(r2_4)

# SVM

sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

svr=SVR(kernel='sigmoid')
svr.fit(x_train,y_train.ravel())
y_pred=svr.predict(x_test)
r2_5=r2_score(y_test,y_pred)
print(r2_5)

# Polinomina Regression

x_ply=PolynomialFeatures(2)
x_train_p=x_ply.fit_transform(x_train)
x_test_p=x_ply.transform(x_test)

lr2=LinearRegression()
lr2.fit(x_train_p,y_train.ravel())
y_pred=lr2.predict(x_test_p)
r2_6=r2_score(y_test,y_pred)

print(r2_6)

'''
##############################################################################

# Classification Task

# Loading Data

data=pd.read_csv('heart.csv')
y=data.iloc[:,-1].values
x=data.drop(['target'],axis=1).values

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33)

sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

# Naive Bayes

gnb=GaussianNB()
gnb.fit(x_train,y_train)

y_pred=gnb.predict(x_test)
cm=confusion_matrix(y_test,y_pred)
acc_sc=accuracy_score(y_test,y_pred)
print('--------------Gaussian Naive Bayes-----------')
print(cm)
print(acc_sc)

# Logistic Regression

lr=LogisticRegressionCV()
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)

cm=confusion_matrix(y_test,y_pred)
acc_sc=accuracy_score(y_test,y_pred)
print('--------------Logistic Regression-----------')
print(cm)
print(acc_sc)

# KNN

knn=KNeighborsClassifier(n_neighbors=6) 
knn.fit(x_train,y_train)
y_pred=knn.predict(x_test)

cm=confusion_matrix(y_test,y_pred)
acc_sc=accuracy_score(y_test,y_pred)
print('--------------K-Nearest Neighbores-----------')
print(cm)
print(acc_sc)

# SVM
# Decision Tree
# Random Forest
# ANN
# CNN





























































# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 17:38:25 2021

@author: Halil İbrahim Bayat

Natural Language Processing

"""

# Essential Libraries for NLP 

import pandas as pd
import numpy as np
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

# Loading Data and Preprocessing

reviews_data=pd.read_csv('r_reviews.csv')
ps=PorterStemmer()
new_data=[]

for i in range(1000):
    reviews=re.sub('[^a-zA-Z]',' ',str(reviews_data['Review'][i]))
    reviews=reviews.lower()
    reviews=reviews.split()
    reviews=[ps.stem(words) for words in reviews if not words in 
             set(stopwords.words('english'))]
    reviews=' '.join(reviews)
    new_data.append(reviews)

# Feature Excraction / Bag of Words / Öznitelik Çıkarımı

cv=CountVectorizer(max_features=2000)
x=cv.fit_transform(new_data).toarray()
y=reviews_data.iloc[:,1].values

#Splitting Data for Test and Train

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20)

# Creating the Model for the Classification

gnb=GaussianNB()
gnb.fit(x_train,y_train.ravel())

# Testing, Confusion Matrix and Calculating Accuracy Rate

y_pred=gnb.predict(x_test)
cm=confusion_matrix(y_test,y_pred)
acc_sc=accuracy_score(y_test,y_pred)
print(cm)
print(f'Accuracy={acc_sc}')






















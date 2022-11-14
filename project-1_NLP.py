# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 18:28:21 2021

@author: Halil İbrahim BAYAT

Natural Languge Processing

"""
# Importing Essential Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Loading Data and Preprocessing

data_train=pd.read_csv('Corona_NLP_train.csv')
data_test=pd.read_csv('Corona_NLP_train.csv')

data_train=data_train.iloc[:,4:7]
data_test=data_test.iloc[:,4:7]

big_data=pd.concat([data_train,data_test],axis=0, ignore_index=True)

ps=PorterStemmer()
data=[]

for i in range(82314):
    data_txt=re.sub('[^a-zA-Z]',' ', str(big_data['OriginalTweet'][i])) # Noktalama işaretlerini çıkarma
    data_txt=data_txt.lower()
    data_txt=data_txt.split()
    data_txt=[ps.stem(words) for words in data_txt if not words in #stopwords'leri eleme
    set(stopwords.words('english'))]
    data_txt =' '.join(data_txt)
    data.append(data_txt)

# Feature Extraction// bag of words

cv=CountVectorizer(max_features=1000)
x=cv.fit_transform(data).toarray()
y=big_data.iloc[:,1].values

# Splitting Data for the test and train

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33)

# Creatin models for classification

KNN=KNeighborsClassifier(n_neighbors=3)
KNN.fit(x_train,y_train)

y_pred=KNN.predict(x_test)

cm=confusion_matrix(y_test,y_pred)

acc_rate=accuracy_score(y_test,y_pred)
print(cm)

print(acc_rate)



















































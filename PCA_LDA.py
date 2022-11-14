# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 16:39:00 2021

@author: Halil İbrahim BAYAT

Boyut İndirgeme ve PCA(Precible Controlling Anylizer)
LDA (Linear Discrimination Analysis)

"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

data=pd.read_csv('wine.csv') 

x=data.iloc[:,0:13].values
y=data.iloc[:,13:].values

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=0)

sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)

pca=PCA(n_components=2)
x_train_2=pca.fit_transform(x_train)
x_test_2=pca.fit_transform(x_test)


# Without PCA

lr=LogisticRegression(random_state=0)
lr.fit(x_train,y_train.ravel())
y_pred=lr.predict(x_test)

cm=confusion_matrix(y_test,y_pred)
acc_score=accuracy_score(y_test,y_pred)
print('--------------Confusion Matrix without PCA----------------------------')
print(cm)
print('------------------------Acc Score Without PCA-------------------------')
print(acc_score)
'''
model=Sequential()

model.add(Dense(13,activation='relu'))
model.add(Dense(26,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy',metrics='accuracy')
model.fit(x_train,y_train,epochs=20)

y_pred=model.predict(x_test)
y_pred=(y_pred>0.5)
cm=confusion_matrix(y_test,y_pred)
acc_score=accuracy_score(y_test,y_pred)
print('No PCA results===>')
print(cm)
print(f'accuracy={acc_score}')
'''

# with PCA


lr2=LogisticRegression(random_state=0)
lr2.fit(x_train_2,y_train.ravel())
y_pred2=lr2.predict(x_test_2)

cm2=confusion_matrix(y_test,y_pred2)
acc_score2=accuracy_score(y_test,y_pred2)
print('--------------Confusion Matrix with PCA-------------------------------')
print(cm)
print('------------------------Acc Score with PCA----------------------------')
print(acc_score)
'''
model2=Sequential()

model2.add(Dense(2,activation='relu'))
model2.add(Dense(13,activation='relu'))
model2.add(Dense(1,activation='sigmoid'))

model2.compile(optimizer='adam', loss='binary_crossentropy',metrics='accuracy')
model2.fit(x_train_2,y_train,epochs=20)

y_pred2=model2.predict(x_test_2)
y_pred2=(y_pred2>0.5)
cm2=confusion_matrix(y_test,y_pred2)
acc_score2=accuracy_score(y_test,y_pred2)
print('No PCA results===>')
print(cm2)
print(f'accuracy={acc_score2}')'''    

# With LDA (Linear Discriminant Analysis)

lda=LDA(n_components=2)

x_train_lda=lda.fit_transform(x_train,y_train.ravel())
x_test_lda=lda.transform(x_test)

lr3=LogisticRegression(random_state=0)
lr3.fit(x_train_lda,y_train.ravel())

y_pred3=lr3.predict(x_test_lda)

cm3=confusion_matrix(y_test,y_pred3)
acc_score3=accuracy_score(y_test,y_pred3)
print('--------------Confusion Matrix with LDA-------------------------------')
print(cm3)
print('------------------------Acc Score with LDA----------------------------')
print(acc_score3)




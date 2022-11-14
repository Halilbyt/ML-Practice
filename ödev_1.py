# -*- coding: utf-8 -*-
# 4.11 ödev uygulaması
"""
Created on Tue Nov  9 12:22:45 2021

@author: Halil İbrahim Bayat
"""
#Gerekli kütüphanelerin yüklenmesi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer 
import statsmodels.api as sm

#Verinin yüklenmesi ve analizler

data=pd.read_csv('tenis.csv')
val1=data.describe()
val2=data.head()
val3=data.corr(method='kendall')

#Nümerik olmayan verilerin dönüşümü

'''Outlook colonunun nümerik formata dönüşümü'''

le=preprocessing.LabelEncoder()
outlook=data.iloc[:,0:1].values

'''outlook[:,0]=le.fit_transform(data.iloc[:,0])
print(outlook)
outlook=data.iloc[:,0:1]
print(outlook)
#Buradaki yaılan işlemler gereksiz ve işimizi görmez
çünkü birden çok değişken var, tek kolonda ifade edilemez 
'''
ohe=preprocessing.OneHotEncoder()
outlook=ohe.fit_transform(outlook).toarray()
print(outlook)

'''Benzer işlemler önce windy sonrada play colonları için yapılıyor'''
windy=data.iloc[:,3:4].values
windy=ohe.fit_transform(windy).toarray()
print(windy)

play=data.iloc[:,4:5].values
play=ohe.fit_transform(play).toarray()

#Bu adımda dönülümleri yapılan colonları tekrar bir data frame olarak bir araya
#getirilecektir. Oncelikle dataframe'e cevrilmeleri gerekmetedir. 

s1=pd.DataFrame(data=outlook,columns=['overcast','rainy','sunny'])
s2=pd.DataFrame(data=windy,columns=['NoWindy','windy'])
s3=pd.DataFrame(data=play, columns=['UnPlayable','playable'])

MyNewData=pd.concat([s1,data.iloc[:,1:3]],axis=1)
MyNewData=pd.concat([MyNewData,s2],axis=1)
myNewData=pd.concat([MyNewData,s3],axis=1)

#Bazı gereksiz stünları atarak nihayi dataframe'i elde etmiş bulunuyoruz

myNewData=myNewData.drop(['NoWindy','UnPlayable'],axis=1)

# Verilerin Ayrılması, başımlı bağımsız değişkenlerin tespiti ve 
# sisteme dahil edilecek veyatta edilmeyecek parametrelerin saptanması

'''Burada bağımlı değişken play colonu olacak ve diğerleri ise bağımsız
değişken olarak sisteme tanıtımını yapacağız.'''
y=myNewData.iloc[:,5:-1].values
x=myNewData.iloc[:,0:6].values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=0)

regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)

# Modelin tekrar düzenlenmesi ve p değerinin bulunması

X=np.append(arr=np.ones((14,1)).astype(int),values=myNewData.iloc[:,:-1],axis=1)
X_1=myNewData.iloc[:,[0,1,2,3,4,5]].values
X_1=np.array(X_1,dtype=float) 

model=sm.OLS(myNewData.iloc[:,-1:],X_1).fit()
print(model.summary())

#windy colonunu atıp tekrar p değerlerini gözlemlemek
#Backward elimination
myNewData=myNewData.drop(['windy'],axis=1)
X=np.append(arr=np.ones((14,1)).astype(int),values=myNewData.iloc[:,:-1],axis=1)
X_1=myNewData.iloc[:,[0,1,2,3,4]].values
X_1=np.array(X_1,dtype=float) 

model=sm.OLS(myNewData.iloc[:,-1:],X_1).fit()
print(model.summary())




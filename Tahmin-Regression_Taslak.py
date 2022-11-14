# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 16:48:03 2021

@author: Halil İbrahim BAYAT
Regresyon Problemeleri, Tahmin Problemleri için Genel Bir Taslak Oluşturulması
ve Bu Taslak Üzerinden Problem Çözümüne Gidilmesi
Buradaki taslağın farkı şu ana kadar çalışılmış bütün metodları ve veri analiz
yöntemlerini kapsamaısıdır.
"""

# Gerekli Kütüphanelerin Yüklenmesi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
import statsmodels.api as sm


# Verinin Yüklenmesi ve incelenmesi

data=pd.read_csv('maaslar_yeni.csv')


# Veride Eksikliklerin Giderilmesi ve Nümerik Olmayan Verilerin Dönüşümü

'''
ulke=data.iloc[:,0:1].values
le=preprocessing.LabelEncoder()
ulke[:,0]=le.fit_transform(data.iloc[:,0])
ulket=data.iloc[:,0:1]

ohe=preprocessing.OneHotEncoder()
ulke=ohe.fit_transform(ulke).toarray()'''


''' veriyi yeniden ölçeklendirmek için: 
sc=StandardScaler()
X_train=sc.fit_transform(x_train) 
X_test=sc.fit_transform(x_test) 
'''
unvan=data.iloc[:,2:3].values
ohe=preprocessing.OneHotEncoder()
unvan=ohe.fit_transform(unvan).toarray()

unvan=pd.DataFrame(data=unvan,columns=['cayci','sekreter','u.yardımcı','uzman','prj.yönt',
                                       'sef','müdür','direktor','c-level','ceo'])
data=data.drop(['Calisan ID','unvan'],axis=1)
data=pd.concat([unvan,data],axis=1)

'''Bağımlı ve bağımsız değişkenkerin belirlenmesi ve atanması'''
x=data.drop(['maas'],axis=1).values
y=data.iloc[:,-1:].values

# Modelin tekrar düzenlenmesi ve p değerinin bulunması

'''X=np.append(arr=np.ones((14,1)).astype(int),values=myNewData.iloc[:,:-1],axis=1)
X_1=myNewData.iloc[:,[0,1,2,3,4,5]].values
X_1=np.array(X_1,dtype=float) 

model=sm.OLS(myNewData.iloc[:,-1:],X_1).fit()
print(model.summary())'''


# Linear Regression
'''linear regresyon modeli olusturma'''

lr=LinearRegression()
lr.fit(x,y)

'''Modelin görsellerini basma:
    
plt.scatter(x,y,color='red')
plt.plot(x,lr.predict(x),color='blue')
    
    '''
model = sm.OLS(lr.predict(x),x)

print(model.fit().summary())
# Polynomia Regression

'''Polinomial Regresyon Modelinin Oluşturulması'''

poly_reg=PolynomialFeatures(degree=2)
x_poly=poly_reg.fit_transform(x)
lr2=LinearRegression()
lr2.fit(x_poly,y)

#poly_r2=r2_score(y,lr2.predict(x))



# Decision Tree Regression

'''Decision Tree Regressor modelinin oluşturulması'''

r_dt=DecisionTreeRegressor()
r_dt.fit(x,y)

r_dt_r2=r2_score(y,r_dt.predict(x))
'''r_dt=DecisionTreeRegressor()
r_dt.fit(x,y)

plt.show()
plt.scatter(x,y)
plt.plot(x,r_dt.predict(x))
'''

# Random Forest Regression

'''Random Forest Regressor Modelinin Oluşturulması'''
rf_reg=RandomForestRegressor(n_estimators=10,random_state=0)
rf_reg.fit(x,y.ravel())
rf_reg_r2=r2_score(y.ravel(),rf_reg.predict(x))
# Support Vector Regression

'''Support Vector Regression modelinin oluşturulması'''

sc1=StandardScaler()
x_scale=sc1.fit_transform(x)
sc2=StandardScaler()
y_scale=sc2.fit_transform(y)
svr_reg=SVR(kernel='rbf')
svr_reg.fit(x_scale,y_scale)

svr_reg_r2=r2_score(y,svr_reg.predict(x))

'''
plt.scatter(x_scale,y_scale, color='red')
plt.plot(x_scale,svr_reg.predict(x_scale),color='black')
'''


# Modellerin R2 hatalarının bulunması ve karşılaştırılması




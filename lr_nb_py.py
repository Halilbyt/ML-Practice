# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 17:57:28 2021

@author: Halil İbrahim BAYAT

Prediction=LinearRegression Methots

Classification=Naive Bayesian Classifier

"""
import numpy as np; import pandas as pd;import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB,BernoulliNB,ComplementNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score,confusion_matrix,classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from string import ascii_uppercase
'''
# Loading Data and Preprocessing

data=pd.read_csv('winequality-red.csv')

corr=data.corr()

# Drawing data

Drawing Correlation Matrix
sb.heatmap(corr, fmt='.1f', annot=True, linewidths=.5)



x=data[['citric acid','sulphates','alcohol']].values
y=data[['quality']].values
sc=StandardScaler()

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.3)

x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)

y_train=sc.fit_transform(y_train)
y_test=sc.fit_transform(y_test)


lr=LinearRegression()

lr.fit(x_train,y_train)

y_pred=lr.predict(x_test)
r2=r2_score(y_test,y_pred)
print('R2 Score of the Linear Regression')
print(r2)

'''

#
# Classification Naive Bayes
# Loading Data and Preprocessing 
#

data=pd.read_csv('heart.csv')

isnull=data.isnull().any()
data_corr=data.corr()



'''
Heatmap drawing both data.corr and saving

corr_heat_m=sb.heatmap(data_corr,annot=True,fmt='.1f')
figure = corr_heat_m.get_figure()    
figure.savefig('corr_heat_map.png', dpi=800)


'''
x=data.drop(['target'],axis=1).values
y=data[['target']].values


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=0)

sc=StandardScaler()
mmsc=MinMaxScaler()

x_train_=mmsc.fit_transform(x_train)
x_test_=mmsc.fit_transform(x_test)

x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)


# Find the best parametrer and train test data sent mixing

gnb=GaussianNB()
gnb.fit(x_train,y_train.ravel())

y_pred=gnb.predict(x_test)

cm=confusion_matrix(y_test,y_pred)
cr=classification_report(y_test,y_pred)
acc_sc=accuracy_score(y_test,y_pred)

results={'Gaussian':acc_sc}

'''
sb.set(font_scale=1.4) # for label size
sb.heatmap(cm, annot=True, annot_kws={"size": 16})

columns = ['class %s' %(i) for i in list(ascii_uppercase)[0:len(np.unique(y_test))]]
cm = pd.DataFrame(cm, index=columns, columns=columns)
sb.heatmap(cm, cmap='Oranges', annot=True)
'''

mnb=MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
mnb.fit(x_train_,y_train.ravel())

y_pred=mnb.predict(x_test_)
acc_sc2=accuracy_score(y_test,y_pred)

results={'Gaussian':acc_sc,'Multinomial':acc_sc2}

#############################################

bnb=BernoulliNB()
bnb.fit(x_train,y_train.ravel())

y_pred=bnb.predict(x_test)
acc_sc3=accuracy_score(y_test,y_pred)

results={'Gaussian':acc_sc,'Multinomial':acc_sc2,'Bernualli':acc_sc3}


#############################################

cnb=ComplementNB()
cnb.fit(x_train_,y_train.ravel())

y_pred=cnb.predict(x_test_)
acc_sc4=accuracy_score(y_test,y_pred)


results={'Gaussian':acc_sc,
         'Multinomial':acc_sc2,
         'Bernualli':acc_sc3,
         'Complement':acc_sc4}

val=str(results.values())
print(val)
results1=pd.DataFrame(results.values(),index=results.keys())

sb.heatmap(results1,annot=True)













































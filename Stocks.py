# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 21:25:27 2020

@author: Win 8
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
df = pd.read_csv('Uniqlo(FastRetailing) 2012-2016 Training - stocks2012-2016.csv')
stock_volatility = abs(((df['Close']-df['Open'])/df['Close'])*100)
df['stock_volatility']= stock_volatility
df

#DATA PREPROCESSING

new_data=df.drop(columns=['Date'])
from sklearn import preprocessing
for i in range(1,len(new_data.columns)+1):
    for j in range(1,len(new_data.index)+1):
        new_data = (new_data-new_data.iloc[i].min())/(new_data.iloc[i].max()-new_data.iloc[i].min())
print(new_data)

#SPLITTING THE DATA INTO TRAINING SET,TEST SET

X=df[['Open','High','Low','Stock Trading','Volume','stock_volatility']]
y=df['Close']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
p =np.arange(len(y_test))

#LINEAR REGRESSION

from sklearn.linear_model import LinearRegression
reg1=LinearRegression()
a = reg1.fit(X_train,y_train)

predict1= a.predict(X_test)
score=reg1.score(X_train,y_train)

df.corr()
sns.heatmap(df.corr())

#DECISION TREE

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
clf=DecisionTreeClassifier()
clf.fit(X_train,y_train)
predict3=clf.predict(X_test)

#RANDOM FOREST

from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators=10)
clf.fit(X_train,y_train)
predict4=clf.predict(X_test)

#CALCULATING ERRORS

predict = [predict1,predict3,predict4]
from sklearn.metrics import mean_absolute_error,mean_squared_error
algorithm = ['Linear regression','Decision Tree','Random Forest']
c=0
for i in predict:
    print(algorithm[c],'\n','Mean Squared Error:',mean_squared_error(y_test,i),'\n','Mean Absolute Error:',mean_absolute_error(y_test,i))
    c+=1
    
"""Linear regression 
 Mean Squared Error: 77730.17414493876 
 Mean Absolute Error: 196.41009297034822
Decision Tree 
 Mean Squared Error: 962757.3170731707 
 Mean Absolute Error: 493.4552845528455
Random Forest 
 Mean Squared Error: 2922957.825203252 
 Mean Absolute Error: 694.7357723577236"""
    
#PLOTTING THE RESULT OF THE BEST ALGORITHM OUT OF THE THREE : LINEAR REGRESSION
    
plt.figure(figsize=(10,6))
plt.figure
plt.plot(p,predict1,color='red',label='predicted price by the model')
plt.plot(p,y_test,color='black',label='Actual price',linestyle='dotted')

plt.legend()
plt.xlabel('Number of days')
plt.ylabel('Stock Trading')
plt.title('STOCK PRICE PREDICTION')
plt.show()


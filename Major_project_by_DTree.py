# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 18:57:51 2021

@author: MY PC
"""

import pandas as pd
df=pd.read_csv("C:/Users/MY PC/Downloads/diabetes.csv")
x=df.iloc[:,1:8].values
y=df.iloc[:,8].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)
from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier()
dtc.fit(x_train,y_train)
y_pred=dtc.predict(x_test)
from sklearn.metrics import accuracy_score,confusion_matrix
accuracy_score( y_test, y_pred)
confusion_matrix( y_test, y_pred)
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 16:53:05 2018

@author: Rishab Teotia
"""

from sklearn.cross_validation import train_test_split
from sklearn import ensemble
from sklearn.preprocessing import Imputer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
data=pd.read_csv('train.csv')

print(data.describe)
print(data.head())

label=data['SalePrice']
#train_data,test_data=data.random_split(.8,seed=0)
train1=data.drop(['Id','SalePrice','SaleCondition','MSZoning','Street','Alley','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','RoofMatl','RoofStyle','Exterior1st','Exterior2nd','MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical','KitchenQual','Functional','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive','PoolQC','Fence','MiscFeature','SaleType','HouseStyle'],axis=1)

#model=linear_model.LinearRegression()
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
train1=imp.fit_transform(train1)
X_Train,X_Test,Y_Train,Y_Test=train_test_split(train1,label,test_size=0.06,random_state=2)
model=ensemble.GradientBoostingRegressor(n_estimators=1200,max_depth=4,min_samples_split=3)
model.fit(X_Train,Y_Train)

model.score(X_Train,Y_Train)
Accuracy=model.score(X_Test,Y_Test)
print('Acurracy of model is:',round(Accuracy*100),"%\n")
predict=model.predict(X_Test)
print("Predicted Price of House is Rs:",predict[0],"\n")
for price in predict:
    print("Predicted Price of House is Rs:",price)
    time.sleep(1)
'''plot=list(range(predict))
plt.bar(plot,predict)
plt.show()
'''

# -*- coding: utf-8 -*-
'''
Created on Mon Apr 27 15:50:20 2020
'''

# =============================================================================
# importing all libraries
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

# =============================================================================
# Reading the data
# =============================================================================

data=pd.read_csv(r'C:\Users\503107711\Desktop\Project for GIT\Bike Rental project\bike_hours_project.csv')
# =============================================================================
# dPrelim Analysis and Feature selection
# =============================================================================

cols=['index','date','casual','registered']
dataset=data.copy()
dataset=dataset.drop(cols,axis=1)

# =============================================================================
# Checking for any missing or null values
# =============================================================================

print(dataset.isnull().sum())

# =============================================================================
# Simple visualization of data using pandas histogram
# =============================================================================
'''
dataset.hist(rwidth=0.9)
plt.tight_layout()
'''
# =============================================================================
# Step 3- Data Visualization
# =============================================================================

''' Visualize demand vs various continous features'''

plt.subplot(2,2,1)
plt.title('Temperature vs Demand')
plt.scatter(dataset.iloc[:,8],dataset.iloc[:,-1],s=2,c='g')
plt.show()

plt.subplot(2,2,2)
plt.title('atemp vs Demand')
plt.scatter(dataset.iloc[:,9],dataset.iloc[:,-1],s=2,c='r')
plt.show()

plt.subplot(2,2,3)
plt.title('Humidity vs Demand')
plt.scatter(dataset.iloc[:,10],dataset.iloc[:,-1],s=2,c='b')
plt.show()

plt.subplot(2,2,4)
plt.title('Windspeed vs Demand')
plt.scatter(dataset.iloc[:,11],dataset.iloc[:,-1],s=2,c='m')
plt.show()

plt.tight_layout()

'''visualizing categorical data'''


colors=['green','blue','red','purple']
plt.subplot(3,3,1)
plt.title('Average Demand per season')
cat_list=dataset['season'].unique()
cat_average=dataset.groupby('season').mean()['demand']

plt.bar(cat_list,cat_average,color=colors)

plt.subplot(3,3,2)
plt.title('Average Demand per year')
cat_list=dataset['year'].unique()
cat_average=dataset.groupby('year').mean()['demand']

plt.bar(cat_list,cat_average,color=colors)

plt.subplot(3,3,3)
plt.title('Average Demand per month')
cat_list=dataset['month'].unique()
cat_average=dataset.groupby('month').mean()['demand']

plt.bar(cat_list,cat_average,color=colors)

plt.subplot(3,3,4)
plt.title('Average Demand per hour')
cat_list=dataset['hour'].unique()
cat_average=dataset.groupby('hour').mean()['demand']

plt.bar(cat_list,cat_average,color=colors)

plt.subplot(3,3,5)
plt.title('Average Demand per holiday')
cat_list=dataset['holiday'].unique()
cat_average=dataset.groupby('holiday').mean()['demand']

plt.bar(cat_list,cat_average,color=colors)

plt.subplot(3,3,6)
plt.title('Average Demand per weekday')
cat_list=dataset['weekday'].unique()
cat_average=dataset.groupby('weekday').mean()['demand']

plt.bar(cat_list,cat_average,color=colors)

plt.subplot(3,3,7)
plt.title('Average Demand per workingday')
cat_list=dataset['workingday'].unique()
cat_average=dataset.groupby('workingday').mean()['demand']

plt.bar(cat_list,cat_average,color=colors)

plt.subplot(3,3,8)
plt.title('Average Demand per weather')
cat_list=dataset['weather'].unique()
cat_average=dataset.groupby('weather').mean()['demand']

plt.bar(cat_list,cat_average,color=colors)

plt.tight_layout()

# =============================================================================
# Checking for Outliers
# =============================================================================

data_quantile=dataset['demand'].quantile([0.05,0.1,0.15,0.9,0.95,0.99])

# =============================================================================
# Check Multiple Linear Regression Assumption
# =============================================================================

''' Linearity using Correlation function matrix using panda'''

correlation_matr=dataset[['temp','atemp','humidity','windspeed','demand']].corr()

'''dropping irrelevant features'''

dataset=dataset.drop(['atemp','weekday','year','workingday','windspeed'],axis=1)

# =============================================================================
# Checking autocorrelation in demand using the acor plot
# =============================================================================
'''values of demand should be of type float for acorr plt'''

dataset1=pd.to_numeric(dataset['demand'],downcast='float')
plt.acorr(dataset1,maxlags=12)
plt.show()

# =============================================================================
# Normalizing the demand feature using log
# =============================================================================


df1=dataset['demand']
df2=np.log(df1)

plt.figure()
plt.hist(df1,rwidth=0.9)
plt.show()

plt.figure()
plt.hist(df2,rwidth=0.9)
plt.show()

''' after with this we see the demand variabl is normally distributed'''


dataset['demand']=np.log(dataset['demand'])

# =============================================================================
# Solving autocorrelation problem in demand feature
# =============================================================================


t_1=dataset['demand'].shift(+1).to_frame()
t_1.columns=['t-1'] 
t_2=dataset['demand'].shift(+2).to_frame()
t_2.columns=['t-2']
t_3=dataset['demand'].shift(+3).to_frame()
t_3.columns=['t-3']

dataset_lag=pd.concat([dataset,t_1,t_2,t_3],axis=1)

dataset_lag=dataset_lag.dropna()


# =============================================================================
# Creating dummy variables for the categorical data
# =============================================================================


dataset_lag['season']=dataset['season'].astype('category')
dataset_lag['month']=dataset['month'].astype('category')
dataset_lag['hour']=dataset['hour'].astype('category')
dataset_lag['holiday']=dataset['holiday'].astype('category')
dataset_lag['weather']=dataset['weather'].astype('category')

dataset_lag=pd.get_dummies(dataset_lag, drop_first=True)


# =============================================================================
# spliting the data into train and test data
# =============================================================================
'''As our data is time depended we cant split by sklearn as the autocorrelation will break'''


X=dataset_lag.drop(['demand'],axis=1)
Y=dataset_lag['demand']

tr_size=0.7*len(X)
tr_size=int(tr_size)

X_train=X.values[0:tr_size]
X_test=X.values[tr_size:len(X)]

Y_train=Y.values[0:tr_size]
Y_test=Y.values[tr_size : len(Y)]

# =============================================================================
# Creating & training the Regression model
# =============================================================================

from sklearn.linear_model import LinearRegression

lr=LinearRegression()

lr.fit(X_train,Y_train)

r_squared1=lr.score(X_train,Y_train)
r_squared2=lr.score(X_test,Y_test)
print(r_squared1)
print(r_squared2)

''' Creating the prediction'''

y_predict=lr.predict(X_test)


from sklearn.metrics import mean_squared_error
root_mse=math.sqrt(mean_squared_error(Y_test,y_predict))

import pickle
filename =r'C:\Users\503107711\Desktop\Project for GIT\Bike Rental project\MLR_project_bike.pkl'
pickle.dump(lr,open(filename,'wb'))



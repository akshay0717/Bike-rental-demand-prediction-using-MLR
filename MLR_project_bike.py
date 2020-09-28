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

def feature_demand_plot(data,feature):
    plt.title(feature + 'Demand')
    plt.scatter(data[feature],data[feature],s=a,c=color)
    

# =============================================================================
# Simple visualization of data using pandas histogram
# =============================================================================
'''
dataset.hist(rwidth=0.9)
plt.tight_layout()
'''
# =============================================================================
# Step 3- Data Visualization
# ====================================================================

''' Visualize demand vs various continous features'''

plt.subplot(2,2,1)
feature_demand_plot(dataset,'Temperature',2,'g')
plt.subplot(2,2,2)
feature_demand_plot(dataset,'atemp',2,'r')
plt.subplot(2,2,3)
feature_demand_plot(dataset,'Humidity',2,'b')
plt.subplot(2,2,4)
feature_demand_plot(dataset,'Windspeed',2,'m')
plt.show()
plt.tight_layout()

'''visualizing categorical data'''


def feature_categorical_plot(data,feature,colors):
    plt.title('Average Demand per '+ feature)
    cat_list=data[feature].unique()
    cat_average=data.groupby(feature).mean()['demand']
    plt.bar(cat_list,cat_average,color=colors)
    
colors=['green','blue','red','purple']

plt.subplot(3,3,1)
feature_categorical_plot(dataset, 'season', colors)
plt.subplot(3,3,2)
feature_categorical_plot(dataset, 'year', colors)
plt.subplot(3,3,3)
feature_categorical_plot(dataset, 'month', colors)
plt.subplot(3,3,4)
feature_categorical_plot(dataset, 'hour', colors)
plt.subplot(3,3,5)
feature_categorical_plot(dataset, 'holiday', colors)
plt.subplot(3,3,6)
feature_categorical_plot(dataset, 'weekday', colors)
plt.subplot(3,3,7)
feature_categorical_plot(dataset, 'workingday', colors)
plt.subplot(3,3,8)
feature_categorical_plot(dataset, 'weather', colors)
plt.show()
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
# Normalizing the demand feature using log transformation
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



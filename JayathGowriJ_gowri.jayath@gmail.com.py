#!/usr/bin/env python
# coding: utf-8

# # Linear Regression model - Sales

# In[97]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings ('ignore')


# In[98]:


df=pd.read_csv('https://raw.githubusercontent.com/training-ml/Files/main/Sales_LinReg.csv')
df.head()


# # Data pre-processing

# In[99]:


df.describe()


# In[100]:


df.isnull().sum()


# In[101]:


# Fill nulls by simple imputer
from sklearn.impute import SimpleImputer

si = SimpleImputer(strategy='mean')

df[['Republic','NDTV','TV9','AajTak']] = si.fit_transform(df[['Republic','NDTV','TV9','AajTak']])
df.isnull().sum()


# Now nulls have been replaced by mean of the respective columns

# In[102]:


df.describe()


# Republic: Std deviation is more when compared to count, so further analysis needs to be done to find outliers. But since its sales data, the higher outlier may not be taken into account as its logically possible.

# # EDA

# In[103]:


# Data distribution to check for skewness
plt.figure(figsize=(15,10))
plotnumber=1
for column in df:
    if plotnumber<=6:
        ax=plt.subplot(2,3,plotnumber)
        sns.distplot(df[column])
        plt.xlabel(column,fontsize=12)
    plotnumber+=1
plt.tight_layout()


# TV5 and TV9 has slight right skewness. It may not significantly impact the performance of your model.

# In[104]:


# Check boxplot for outliers
features=df[['TV5','TV9']]
plt.figure(figsize=(10,15))
graph=1
for column in features:
    if graph<=7:
        plt.subplot(3,3,graph)
        ax=sns.boxplot(data=features[column])
        plt.xlabel(column, fontsize=15)
    graph+=1
plt.show()


# Only few outliers present as per boxplot.

# In[105]:


# Observe relationship between all features and label
y=df['sales']
x=df.drop(columns=['sales'])#OR select only necessary columns x=data[['GRE Score','TOEFL Score','University Rating' ]]
plt.figure(figsize=(12,9),facecolor='pink')
plotnumber=1
for column in x:
    if plotnumber<=8:
        ax=plt.subplot(2,4,plotnumber)
        plt.scatter(x[column],y)
        plt.xlabel(column)
        plt.ylabel('Sales')
    plotnumber+=1
plt.tight_layout()


# Republic, NDTV, and AajTak shows significant positive linear relationship with sales.  
# TV9 and TV5 shows somewhat positive linear relationship with sales. 

# # FEATURE SELECTION

# In[106]:


# Check for multicollinearity using pairplot (Find relationship between variables- Qualitative assessment)
x = df.drop(columns=['sales'], axis=1)

plt.figure(figsize=(15, 8))
sns.pairplot(x)
plt.suptitle("Pairplot to Check Multicollinearity", y=1.02)
plt.show()


# In[132]:


x = df[['NDTV', 'AajTak']]

plt.figure(figsize=(6, 4))
plt.scatter(x['NDTV'], x['AajTak'], c='blue', alpha=0.5)  # Adjust color and transparency as needed
plt.title('Scatter Plot of NDTV vs AajTak')

plt.show()


# There is positive correlation between most of the news channels.   
# The correlation between Republic and NDTV is stronger, so use vif to confirm multicollinearity.

# In[107]:


# Check for multicollinearity using variance inflation factor (VIF) calculations.
from statsmodels.stats.outliers_influence import variance_inflation_factor

x = df.drop(columns=['sales'], axis=1)
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

vif = pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(x_scaled, i) for i in range(x_scaled.shape[1])]
vif['Features'] = x.columns
vif


# As vif of NDTV is higher, it might be better to remove it as it avoids overfitting.

# In[108]:


df1=df.drop(columns='NDTV', axis=1)
df1.head()


# Selected features = Republic, TV5, TV9, AajTak

# # Model building

# In[109]:


x = df1.drop(columns=['sales'], axis=1)
y = df1['sales']

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=58)


# In[110]:


reg=LinearRegression()
reg.fit(x_train,y_train)


# In[111]:


df1.tail()


# In[112]:


# predict for individual set
reg.predict(scaler.transform([[40,18,6,18]]))


# Prediction is not much accurate as it gave 7.9 instead of 10.8

# # Model evaluation

# In[113]:


# Check score
a=reg.score(x_train,y_train)
print('Training score : ',a)
b=reg.score(x_test,y_test)
print('Testing score : ',b)


# There is difference in Training score and Testing score and Testing score is lesser.

# In[114]:


y_pred=reg.predict(x_test)


# In[115]:


# Check MAE, MSE and RMSE of the model
from sklearn.metrics import mean_absolute_error, mean_squared_error
a = mean_absolute_error(y_test,y_pred)
b=mean_squared_error(y_test,y_pred)
c=np.sqrt(mean_squared_error(y_test,y_pred))
print('\nMAE =', a)
print('MSE =', b)
print('RMSE =', c)


# In[116]:


# Regularization using RidgeCV
from sklearn.linear_model import Lasso, LassoCV, Ridge, RidgeCV
lassocv=LassoCV(alphas=None, max_iter=10)
lassocv.fit(x_train,y_train)
alpha=lassocv.alpha_
lassreg=Lasso(alpha)
lassreg.fit(x_train,y_train)
lassreg.score(x_test,y_test)


# In[117]:


ridgecv = RidgeCV(alphas=np.arange(0.001, 0.1, 0.01))
ridgecv.fit(x_train, y_train)
rid_alpha = ridgecv.alpha_
ridreg = Ridge(alpha=rid_alpha)
ridreg.fit(x_train, y_train)
ridreg.score(x_test, y_test)


# Testing score is 69.5% and regularization has not altered it.

# # Since the score is less and prediction is not accurate, we could try adding the removed feature i.e., NDTV afor model building and check the score and individual prediction for improvement.

# In[119]:


df.tail()


# In[120]:


x = df.drop(columns=['sales'], axis=1)
y = df['sales']

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=58)


# In[121]:


reg1=LinearRegression()
reg1.fit(x_train,y_train)


# In[122]:


reg1.predict(scaler.transform([[40,40,18,6,18]]))


# Prediction has comparitively improved as it gave 11.9 now instead of 10.8

# In[123]:


# Check score
a=reg1.score(x_train,y_train)
print('Training score : ',a)
b=reg1.score(x_test,y_test)
print('Testing score : ',b)


# In[125]:


y_pred=reg1.predict(x_test)


# In[126]:


# Check MAE, MSE and RMSE of the model
from sklearn.metrics import mean_absolute_error, mean_squared_error
a = mean_absolute_error(y_test,y_pred)
b=mean_squared_error(y_test,y_pred)
c=np.sqrt(mean_squared_error(y_test,y_pred))
print('\nMAE =', a)
print('MSE =', b)
print('RMSE =', c)


# In[127]:


# Regularization using LassoCV
from sklearn.linear_model import Lasso, LassoCV, Ridge, RidgeCV
lassocv=LassoCV(alphas=None, max_iter=10)
lassocv.fit(x_train,y_train)
alpha=lassocv.alpha_
lassreg=Lasso(alpha)
lassreg.fit(x_train,y_train)
lassreg.score(x_test,y_test)


# In[128]:


# Regularization using RidgeCV
ridgecv = RidgeCV(alphas=np.arange(0.001, 0.1, 0.01))
ridgecv.fit(x_train, y_train)
rid_alpha = ridgecv.alpha_
ridreg = Ridge(alpha=rid_alpha)
ridreg.fit(x_train, y_train)
ridreg.score(x_test, y_test)


# # Now the model score = 81% and regularization has not altered the score. Individual prediction has also given nearest sales value. Therefore, by evaluation of scores, even though there was correlation between NDTV and AajTak, removal of features seemed unnecessary.

# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# # Task6: Predicting Real Estate House Prices

# ## This task is provided to test your understanding of building a Linear Regression model for a provided dataset

# ### Dataset: Real_estate.csv

# ### Import the necessary libraries
# #### Hint: Also import seaborn

# In[24]:


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


# ### Read the csv data into a pandas dataframe and display the first 5 samples

# In[3]:


data = pd.read_csv('Real estate.csv')
data


# ### Show more information about the dataset

# In[4]:


data.info()


# ### Find how many samples are there and how many columns are there in the dataset

# In[6]:


data.shape


# ### What are the features available in the dataset?

# In[10]:


data.columns


# ### Check if any features have missing data

# In[11]:


data.isna().sum()


# ### Group all the features as dependent features in X

# In[16]:


X = data.iloc[:,:-1]
X


# ### Group feature(s) as independent features in y

# In[14]:


y = data.iloc[:,-1]
y


# ### Split the dataset into train and test data

# In[62]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = .75)


# ### Choose the model (Linear Regression)

# In[63]:


from sklearn.linear_model import LinearRegression


# ### Create an Estimator object

# In[64]:


est = LinearRegression()


# ### Train the model

# In[65]:


est.fit(X_train, y_train)


# ### Apply the model

# In[78]:


y_pred = est.predict(X_test)


# ### Display the coefficients

# In[79]:


est.coef_


# ### Find how well the trained model did with testing data

# In[80]:


print('r2:', r2_score(y_test, y_pred))


# ### Plot House Age Vs Price
# #### Hint: Use regplot in sns

# In[73]:


sns.regplot(x="X2 house age", y="Y house price of unit area", data=data)


# ### Plot Distance to MRT station Vs Price

# In[74]:


sns.regplot(y="X3 distance to the nearest MRT station", x="Y house price of unit area", data=data)


# ### Plot Number of Convienience Stores Vs Price

# In[75]:


sns.regplot(y="X4 number of convenience stores", x="Y house price of unit area", data=data)


# In[ ]:





# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# # Task 10 : Benchmark Top ML Algorithms
# 
# This task tests your ability to use different ML algorithms when solving a specific problem.
# 

# ### Dataset
# Predict Loan Eligibility for Dream Housing Finance company
# 
# Dream Housing Finance company deals in all kinds of home loans. They have presence across all urban, semi urban and rural areas. Customer first applies for home loan and after that company validates the customer eligibility for loan.
# 
# Company wants to automate the loan eligibility process (real time) based on customer detail provided while filling online application form. These details are Gender, Marital Status, Education, Number of Dependents, Income, Loan Amount, Credit History and others. To automate this process, they have provided a dataset to identify the customers segments that are eligible for loan amount so that they can specifically target these customers.
# 
# Train: https://raw.githubusercontent.com/subashgandyer/datasets/main/loan_train.csv
# 
# Test: https://raw.githubusercontent.com/subashgandyer/datasets/main/loan_test.csv

# ## Task Requirements
# ### You can have the following Classification models built using different ML algorithms
# - Decision Tree
# - KNN
# - Logistic Regression
# - SVM
# - Random Forest
# - Any other algorithm of your choice

# ### Use GridSearchCV for finding the best model with the best hyperparameters

# - ### Build models
# - ### Create Parameter Grid
# - ### Run GridSearchCV
# - ### Choose the best model with the best hyperparameter
# - ### Give the best accuracy
# - ### Also, benchmark the best accuracy that you could get for every classification algorithm asked above

# #### Your final output will be something like this:
# - Best algorithm accuracy
# - Best hyperparameter accuracy for every algorithm
# 
# **Table 1 (Algorithm wise best model with best hyperparameter)**
# 
# Algorithm   |     Accuracy   |   Hyperparameters
# - DT
# - KNN
# - LR
# - SVM
# - RF
# - anyother
# 
# **Table 2 (Best overall)**
# 
# Algorithm    |   Accuracy    |   Hyperparameters
# 
# 

# ### Submission
# - Submit Notebook containing all saved ran code with outputs
# - Document with the above two tables

# In[7]:


train_url = 'https://raw.githubusercontent.com/subashgandyer/datasets/main/loan_train.csv'
test_url = 'https://raw.githubusercontent.com/subashgandyer/datasets/main/loan_test.csv'


# In[8]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt
import seaborn as sns


# In[9]:


train = pd.read_csv(train_url)
test = pd.read_csv(test_url)


# In[10]:


train.head()


# In[11]:


train.dtypes


# In[12]:


print(train.isna().sum())


# In[13]:


print(test.isna().sum())


# In[14]:


train.shape


# In[15]:


test_y = train['Loan_Status']
train = train.drop(['Loan_Status', 'Loan_ID'], axis=1)


# In[16]:


cat_train = train[['Gender', 'Dependents','Married', 'Self_Employed', 'Education', 'Property_Area']]

cat_test = test[['Gender', 'Dependents', 'Married', 'Self_Employed', 'Education', 'Property_Area']]


# In[17]:


print(cat_test.columns)


# In[18]:


print(cat_train.columns)


# In[19]:


num_test = test[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']]
num_train = train[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']]


# In[20]:


test.Gender.value_counts()


# In[23]:


cat_test['Gender'].fillna('Male', inplace=True)
cat_train['Gender'].fillna('Male', inplace=True)
cat_test.Gender.value_counts()


# In[22]:


cat_test.Self_Employed.value_counts()


# In[24]:


cat_test['Self_Employed'].fillna('No', inplace=True)
cat_train['Self_Employed'].fillna('No', inplace=True)
cat_test.Self_Employed.value_counts()


# In[25]:


cat_test.Dependents.value_counts()


# In[26]:


cat_train['Dependents'].fillna("0", inplace=True)
cat_test['Dependents'].fillna("0", inplace=True)
cat_test.Dependents.value_counts()


# In[27]:


cat_test.Married.value_counts()


# In[28]:


cat_train['Married'].fillna("Yes", inplace=True)
cat_test['Married'].fillna("Yes", inplace=True)
cat_test.Married.value_counts()


# In[29]:


cat_test.Self_Employed.value_counts()


# In[30]:


cat_train['Self_Employed'].fillna("No", inplace=True)
cat_test['Self_Employed'].fillna("No", inplace=True)
cat_test.Self_Employed.value_counts()


# In[31]:


plt.figure(figsize=(10,6))
sns.heatmap(cat_train.isna(), cbar=False, cmap='viridis', yticklabels=False)


# In[32]:


plt.figure(figsize=(10,6))
sns.heatmap(cat_test.isna(), cbar=False, cmap='viridis', yticklabels=False)


# In[33]:


print(num_train.isna().sum())


# In[34]:


print(num_test.isna().sum())


# In[35]:


from sklearn.impute import SimpleImputer


# In[36]:


imputer = SimpleImputer(strategy='mean')


# In[37]:


imputer.fit(num_test)


# In[38]:


num_test_transform = imputer.transform(num_test)


# In[39]:


imputer.fit(num_train)


# In[40]:


num_train_transform = imputer.transform(num_train)


# In[41]:


num_test = pd.DataFrame(data=num_test_transform)
num_test.columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
       'Loan_Amount_Term', 'Credit_History']
num_test


# In[42]:


num_train = pd.DataFrame(data=num_train_transform)
num_train.columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
       'Loan_Amount_Term', 'Credit_History']
num_train


# In[43]:


from sklearn.preprocessing import LabelEncoder


# In[44]:


label_encoder = LabelEncoder()


# In[45]:


def clean_dep(x):
    return x[0]


# In[46]:


print(cat_test.isna().sum())


# In[47]:


cat_train['Gender']= label_encoder.fit_transform(cat_train['Gender'])
cat_train['Married']= label_encoder.fit_transform(cat_train['Married'])
cat_train['Self_Employed']= label_encoder.fit_transform(cat_train['Self_Employed'])
cat_train['Education']= label_encoder.fit_transform(cat_train['Education']) 
cat_train['Property_Area']= label_encoder.fit_transform(cat_train['Education']) 
cat_train['Dependents']= label_encoder.fit_transform(cat_train['Dependents'])
cat_train


# In[48]:


plt.figure(figsize=(10,6))
sns.heatmap(cat_train.isna(), cbar=False, cmap='viridis', yticklabels=False)


# In[49]:


cat_test['Gender']= label_encoder.fit_transform(cat_test['Gender'])
cat_test['Married']= label_encoder.fit_transform(cat_test['Married'])
cat_test['Self_Employed']= label_encoder.fit_transform(cat_test['Self_Employed'])
cat_test['Education']= label_encoder.fit_transform(cat_test['Education']) 
cat_test['Property_Area']= label_encoder.fit_transform(cat_test['Education']) 
cat_test['Dependents']= label_encoder.fit_transform(cat_test['Dependents'])
cat_test


# In[50]:


plt.figure(figsize=(10,6))
sns.heatmap(cat_test.isna(), cbar=False, cmap='viridis', yticklabels=False)


# In[51]:


test_y.value_counts()


# In[52]:


test_y= label_encoder.fit_transform(test_y)


# In[53]:


test_label = pd.DataFrame(test_y, columns=['Loan_Status'])


# In[54]:


test_label.value_counts()


# In[55]:


df_test = pd.concat([cat_test, num_test], axis=1)
df_train = pd.concat([cat_train, num_train], axis=1)


# In[56]:


df_test.head()


# In[57]:


plt.figure(figsize=(10,6))
sns.heatmap(df_test.isna(), cbar=False, cmap='viridis', yticklabels=False)


# In[1]:


############################# Model Building ##############################


# In[81]:


def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


# In[90]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from numpy import mean
from numpy import std
from numpy import arange
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV 
from pprint import pprint
from sklearn.pipeline import Pipeline


# In[76]:


x_tn, x_val, y_tn, y_val = train_test_split(df_train, 
                                            test_y, 
                                            test_size=0.33, 
                                            random_state=1
)


# In[88]:


# Created differing hyperparameters that were as simple as possible, 
# demonstrating that the method works and can be extrapolated to more time consuming hyperparameters
pipe = Pipeline([("classifier", LogisticRegression())])

search_space = [
                {"classifier": [LogisticRegression()],
                 "classifier__fit_intercept": [True, False]
                 },
                {"classifier": [KNeighborsClassifier()],
                 "classifier__algorithm": ['auto', 'ball']
                 },
                {"classifier": [DecisionTreeClassifier()],
                 "classifier__criterion": ['gini', 'entropy']
                 },
                {"classifier": [SVC()],
                 "classifier__kernel": ['linear', 'rbf']
                 },
                {"classifier": [GaussianNB()],
                 "classifier__var_smoothing": ['1e-9', 1]
                 },
                {"classifier": [RandomForestClassifier()],
                 "classifier__n_estimators": [100, 120]}]


# In[91]:


gridsearch = GridSearchCV(pipe, search_space, cv=5, verbose=0,n_jobs=-1) # Fit grid search
best_model = gridsearch.fit(x_tn, y_tn)


# In[94]:


print(best_model.best_estimator_)


# In[95]:


rfc_random = print("The mean accuracy of the model is:",best_model.score(x_tn, y_tn))


# In[122]:


gridsearch.cv_results_


# In[131]:


results = gridsearch.cv_results_
model_list = np.array(results['param_classifier'])
model_list = pd.DataFrame(model_list, columns=['Model'])
model_list


# In[142]:


results_list = np.array(results['mean_test_score'])
results_list = pd.DataFrame(results_list, columns=['Score'])
Result = pd.concat([model_list, results_list], axis=1)
Result = Result.sort_values(by='Score',ascending=False)
Result = Result.drop_duplicates(subset='Model', keep='first', inplace=False)
Result


# In[ ]:





# In[119]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





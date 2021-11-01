#!/usr/bin/env python
# coding: utf-8

# ## IterativeImputer
# ### This notebook outlines the usage of Iterative Imputer (Multivariate Imputation).
# ### Iterative Imputer substitutes missing values as a function of other features
# #### Dataset: [https://github.com/subashgandyer/datasets/blob/main/heart_disease.csv]

# **Demographic**
# - Sex: male or female(Nominal)
# - Age: Age of the patient;(Continuous - Although the recorded ages have been truncated to whole numbers, the concept of age is continuous)
# 
# **Behavioral**
# - Current Smoker: whether or not the patient is a current smoker (Nominal)
# - Cigs Per Day: the number of cigarettes that the person smoked on average in one day.(can be considered continuous as one can have any number of cigarettes, even half a cigarette.)
# 
# **Medical(history)**
# - BP Meds: whether or not the patient was on blood pressure medication (Nominal)
# - Prevalent Stroke: whether or not the patient had previously had a stroke (Nominal)
# - Prevalent Hyp: whether or not the patient was hypertensive (Nominal)
# - Diabetes: whether or not the patient had diabetes (Nominal)
# 
# **Medical(current)**
# - Tot Chol: total cholesterol level (Continuous)
# - Sys BP: systolic blood pressure (Continuous)
# - Dia BP: diastolic blood pressure (Continuous)
# - BMI: Body Mass Index (Continuous)
# - Heart Rate: heart rate (Continuous - In medical research, variables such as heart rate though in fact discrete, yet are considered continuous because of large number of possible values.)
# - Glucose: glucose level (Continuous)
# 
# **Predict variable (desired target)**
# - 10 year risk of coronary heart disease CHD (binary: “1”, means “Yes”, “0” means “No”)

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


# In[2]:


url = 'https://raw.githubusercontent.com/subashgandyer/datasets/main/heart_disease.csv'
df=pd.read_csv(url)
df


# ### How many Categorical variables in the dataset?

# In[16]:


df.info()


# ### How many Missing values in the dataset?
# Hint: df.Series.isna( ).sum( )

# In[3]:


for i in range(len(df.columns)):
    missing_data = df[df.columns[i]].isna().sum()
    perc = missing_data / len(df) * 100
    print(f'Feature {i+1} >> Missing entries: {missing_data}  |  Percentage: {round(perc, 2)}')


# ### Bonus: Visual representation of missing values

# In[4]:


plt.figure(figsize=(10,6))
sns.heatmap(df.isna(), cbar=False, cmap='viridis', yticklabels=False)


# ### Import IterativeImputer

# In[5]:


from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


# ### Create IterativeImputer object with max_iterations and random_state=0

# In[6]:


imputer = IterativeImputer(max_iter=10, random_state=0)


# ### Optional - converting df into numpy array

# In[7]:


data = df.values


# In[8]:


X = data[:, :-1]
y = data[:, -1]


# ### Fit the imputer model on dataset to perform iterative multivariate imputation

# In[9]:


imputer.fit(X)


# ### Trained imputer model is applied to dataset to create a copy of dataset with all filled missing values using transform( ) 

# In[10]:


X_transform = imputer.transform(X)


# ### Sanity Check: Whether missing values are filled or not

# In[11]:


print(f"Missing cells: {sum(np.isnan(X).flatten())}")


# In[12]:


print(f"Missing cells: {sum(np.isnan(X_transform).flatten())}")


# ### Let's try to visualize the missing values.

# In[13]:


plt.figure(figsize=(10,6))
sns.heatmap(df.isna(), cbar=False, cmap='viridis', yticklabels=False)


# In[14]:


plt.figure(figsize=(10,6))
sns.heatmap(X_transform.isna(), cbar=False, cmap='viridis', yticklabels=False)


# ### What's the issue here?
# #### Hint: Heatmap needs a DataFrame and not a Numpy Array

# In[15]:


df_transform = pd.DataFrame(data=X_transform)
df_transform


# In[16]:


plt.figure(figsize=(10,6))
sns.heatmap(df_transform.isna(), cbar=False, cmap='viridis', yticklabels=False)


# # Check if these datasets contain missing data
# ### Load the datasets

# In[17]:


X_train = pd.read_csv("X_train.csv")
Y_train = pd.read_csv("Y_train.csv")
Y_test = pd.read_csv("Y_test.csv")
X_test = pd.read_csv("X_test.csv")


# In[18]:


X_train.shape, Y_train.shape, X_test.shape, Y_test.shape


# In[19]:


plt.figure(figsize=(10,6))
sns.heatmap(X_train.isna(), cbar=False, cmap='viridis', yticklabels=False)


# ### Is there missing data in this dataset???

# In[20]:


#No


# # Build a Logistic Regression model Without imputation

# In[21]:


url = 'https://raw.githubusercontent.com/subashgandyer/datasets/main/heart_disease.csv'
df=pd.read_csv(url)
X = df[df.columns[:-1]]
y = df[df.columns[-1]]


# In[22]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[23]:


model = LogisticRegression()


# In[24]:


model.fit(X,y)


# # Drop all rows with missing entries - Build a Logistic Regression model and benchmark the accuracy

# In[25]:


from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score


# In[26]:


url = 'https://raw.githubusercontent.com/subashgandyer/datasets/main/heart_disease.csv'
df=pd.read_csv(url)
df


# In[27]:


df.shape


# ### Drop rows with missing values

# In[28]:


df = df.dropna()
df.shape


# ### Split dataset into X and y

# In[29]:


X = df[df.columns[:-1]]
X.shape


# In[30]:


y = df[df.columns[-1]]
y.shape


# ### Create a pipeline with model parameter

# In[31]:


pipeline = Pipeline([('model', model)])


# ### Create a RepeatedStratifiedKFold with 10 splits and 3 repeats and random_state=1

# In[32]:


cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)


# ### Call cross_val_score with pipeline, X, y, accuracy metric and cv

# In[33]:


scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1)


# In[34]:


scores


# ### Print the Mean Accuracy and Standard Deviation from scores

# In[35]:


print(f"Mean Accuracy: {round(np.mean(scores), 3)}  | Std: {round(np.std(scores), 3)}")


# # Build a Logistic Regression model with IterativeImputer

# In[36]:


from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score


# In[37]:


url = 'https://raw.githubusercontent.com/subashgandyer/datasets/main/heart_disease.csv'
df=pd.read_csv(url)
df


# ### Split dataset into X and y

# In[38]:


df.shape


# In[39]:


X = df[df.columns[:-1]]
X.shape


# In[40]:


y = df[df.columns[-1]]
y


# ### Create a SimpleImputer with mean strategy

# In[41]:


imputer = IterativeImputer(max_iter=10, random_state=0)


# ### Create a Logistic Regression model

# In[42]:


model = LogisticRegression()


# ### Create a pipeline with impute and model parameters

# In[43]:


pipeline = Pipeline([('impute', imputer), ('model', model)])


# ### Create a RepeatedStratifiedKFold with 10 splits and 3 repeats and random_state=1

# In[44]:


cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)


# ### Call cross_val_score with pipeline, X, y, accuracy metric and cv

# In[45]:


scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1)


# In[46]:


scores


# ### Print the Mean Accuracy and Standard Deviation

# In[47]:


print(f"Mean Accuracy: {round(np.mean(scores), 3)}  | Std: {round(np.std(scores), 3)}")


# ### Which accuracy is better? 
# - Dropping missing values
# - SimpleImputer with Mean Strategy

# In[ ]:


#Iterative imputer increased the accuracy and appears to have been better


# # IterativeImputer with RandomForest

# In[48]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.impute import SimpleImputer
from numpy import mean
from numpy import std
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.linear_model import LogisticRegression


# In[49]:


imputer = IterativeImputer(max_iter=10, random_state=0)


# In[50]:


model = RandomForestClassifier()


# In[51]:


pipeline = Pipeline([('impute', imputer), ('model', model)])


# In[52]:


cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)


# In[53]:


scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1)


# In[54]:


print(f"Mean Accuracy: {round(np.mean(scores), 3)}  | Std: {round(np.std(scores), 3)}")


# # Run experiments with different Imputation methods and different algorithms
# 
# ## Imputation Methods
# - Mean
# - Median
# - Most_frequent
# - Constant
# - IterativeImputer
# 
# ## ALGORITHMS
# - Logistic Regression
# - KNN
# - Random Forest
# - SVM
# - Any other algorithm of your choice

# In[56]:


results =[]
model = RandomForestClassifier
strategies = ['mean', 'median', 'most_frequent','constant']

for s in strategies:
    pipeline = Pipeline([('impute', SimpleImputer(strategy=s)),('model', model)])
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    results.append(scores)

for method, accuracy in zip(strategies, results):
    print(f"Strategy: {method} >> Accuracy: {round(np.mean(accuracy), 3)}   |   Max accuracy: {round(np.max(accuracy), 3)}")
          
          


# # Q1: Which is the best strategy for this dataset using Random Forest algorithm?
# - SimpleImputer(Mean)
# - SimpleImputer(Median)
# - SimpleImputer(Most_frequent)
# - SimpleImputer(Constant)
# - IterativeImputer

# In[87]:


model = RandomForestClassifier()
results = []
imputer = [SimpleImputer(strategy='mean'), SimpleImputer(strategy='median'), SimpleImputer(strategy='most_frequent'), SimpleImputer(strategy='constant'), IterativeImputer()]

for s in imputer:
    pipeline = Pipeline(steps=[('i', s), ('m', model)])
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    results.append(scores)

for imputer, results in zip(imputer, results):
    print(f"Strategy: {imputer} >> Accuracy: {round(np.mean(accuracy), 3)}   |   Max accuracy: {round(np.max(accuracy), 3)}")

#The best dataset appears to be a tie between SimpleInputer using mean and iterative imputer.
# # Q2:  Which is the best algorithm for this dataset using IterativeImputer?
# - Logistic Regression
# - Random Forest
# - KNN
# - any other algorithm of your choice (BONUS)

# In[68]:


results =[]
models = [LogisticRegression(), RandomForestClassifier(), KNN(), DecisionTreeClassifier()]

for s in models:
    pipeline = Pipeline(steps=[('i', IterativeImputer(max_iter=10)), ('m', s)])
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    results.append(scores)

for models, accuracy in zip(models, results):
    print(f"Strategy: {models} >> Accuracy: {round(np.mean(accuracy), 3)}   |   Max accuracy: {round(np.max(accuracy), 3)}")


# In[ ]:


#The best model appears to be logistic Regression.


# # Q3: Which is the best combination of algorithm and best Imputation Strategy overall?
# - Mean , Median, Most_frequent, Constant, IterativeImputer
# - Logistic Regression, Random Forest, KNN

# In[85]:


results =[]
models = [LogisticRegression(), RandomForestClassifier(), KNN(), DecisionTreeClassifier()]
strategies = [SimpleImputer(strategy='mean'), SimpleImputer(strategy='median'), SimpleImputer(strategy='most_frequent'), SimpleImputer(strategy='constant'), IterativeImputer()]

for a in models:
    for s in strategies:
        pipeline = Pipeline(steps=[('i', s), ('m', a)])
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
        results.append(scores)
        print('>%s %s %.3f (%.3f)' % (a, s, mean(scores), np.max(scores)))


# In[ ]:


#The most accurate method appears to be a tie between LogisticReggression mean and RandomForest mean


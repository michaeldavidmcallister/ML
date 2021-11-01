#!/usr/bin/env python
# coding: utf-8

# # Task 8 - Find the best Random Forest through Random Search
# 
# In order to **maximize the performance of the random forest**, we can perform a **random search** for better hyperparameters. This will randomly select combinations of hyperparameters from a grid, evaluate them using cross validation on the training data, and return the values (read best model with hyperparameters) that perform the best. 

# ### Task Requirements
# - Build a RandomForest for the above dataset (not one but many with different sets of parameters)
# - Explore RandomizedSearchCV in Scikit-learn documentation
# - Create a parameter grid with these values
#     - n_estimators : between 10 and 200
#     - max_depth : choose between 3 and 20
#     - max_features : ['auto', 'sqrt', None] + list(np.arange(0.5, 1, 0.1))
#     - max_leaf_nodes : choose between 10 to 50
#     - min_samples_split : choose between 2, 5, or 10
#     - bootstrap : choose between True or False
# - Create the estimator (RandomForestClassifier)
# - Create the RandomizedSearchCV with estimator, parameter grid, scoring on roc auc, n_iter = 10, random_state=RSEED(50) for same reproducible results
# - Fit the model
# - Explore the best model parameters
# - Use the best model parameters to predict
# - Plot the best model ROC AUC Curve
# - Plot the Confusion Matrix
# - Write any insights or observations you found in the last

# ## Random Forest Theory revisited

# ### Random Forest = Decision Tree + Bagging + Random subsets of features

# The Random Forest is a model made up of many `decision trees`. Rather than just simply averaging the prediction of trees (which we could call a **forest**), this model uses two key concepts that gives it the name random:
# - Random sampling of training data points when building trees
# - Random subsets of features considered when splitting nodes

# To be more clear, this takes the idea of a single decision tree, and creates an _ensemble_ model out of hundreds or thousands of trees to reduce the variance. 
# 
# Each tree is trained on a random set of the observations, and for each split of a node, only a `subset of the features` are used for making a split. When making predictions, the random forest `averages the predictions` for each of the individual decision trees for each data point in order to arrive at a final classification.

# ### Bagging
# 
# ### Random sampling of training observations
# 
# - **Training**: each tree in a random forest learns from a **random sample** of the data points. The samples are drawn with replacement, known as **bootstrapping**, which means that some samples will be used multiple times in a single tree. The idea is that by training each tree on different samples, although each tree might have high variance with respect to a particular set of the training data, overall, the entire forest will have lower variance but not at the cost of increasing the bias.
# 
# - **Testing**: predictions are made by **averaging the predictions** of each decision tree. This procedure of training each individual learner on different bootstrapped subsets of the data and then averaging the predictions is known as **bagging**, short for **bootstrap aggregating**.

# ### Random Subsets of features for splitting nodes
# Only a subset of all the features are considered for splitting each node in each decision tree. Generally this is set to `sqrt(n_features)` for classification meaning that if there are 16 features, at each node in each tree, only 4 random features will be considered for splitting the node. 

# ### Let us see if our theory holds good in the same dataset we used for building Decision Tree

# In[ ]:





# # Behavioral Risk Factor Surveillance System
# 
# [Behavioral Risk Factor Surveillance System](https://www.kaggle.com/cdc/behavioral-risk-factor-surveillance-system)
# 
# The objective of the BRFSS is to collect uniform, state-specific data on preventive health practices and risk behaviors that are linked to chronic diseases, injuries, and preventable infectious diseases in the adult population. Factors assessed by the BRFSS include tobacco use, health care coverage, HIV/AIDS knowledge or prevention, physical activity, and fruit and vegetable consumption. Data are collected from a random sample of adults (one per household) through a telephone survey.
# 
# The Behavioral Risk Factor Surveillance System (BRFSS) is the nation's premier system of health-related telephone surveys that collect state data about U.S. residents regarding their health-related risk behaviors, chronic health conditions, and use of preventive services. Established in 1984 with 15 states, BRFSS now collects data in all 50 states as well as the District of Columbia and three U.S. territories. BRFSS completes more than 400,000 adult interviews each year, making it the largest continuously conducted health survey system in the world.
# 
# The following data set is from the Centers for Disease Control and Prevention (CDC) and includes socioeconomic and lifestyle indicators for hundreds of thousands of individuals. The objective is to predict the overall health of an individual: either 0 for poor health or 1 for good health. We'll limit the data to 100,000 individuals to speed up training.
# 
# Or, if you have the gut to take it, please pass the entire data and have fun!!!
# 
# This problem is imbalanced (far more of one label than another) so for assessing performance, we'll use recall, precision, receiver operating characteristic area under the curve (ROC AUC), and also plot the ROC curve. Accuracy is not a useful metric when dealing with an imbalanced problem. **Why????**

# ## Data Acquisition
# Go to Kaggle Competition page and pull the dataset of 2015

# In[2]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
RSEED=50


# In[4]:


df = pd.read_csv('2015.csv').sample(100000, random_state = RSEED)
df.head()


# ### Data Exploration
# - Find how many features
# - Find how many samples
# - Find how many missing data
# - Find how many categorical features
# - And many more

# In[5]:


df = df.select_dtypes('number')
df


# ### Label Distribution
# RFHLTH is the label for this dataset

# ### Explore the label

# In[6]:


df['_RFHLTH']


# ### Find what are the values inside the label

# In[7]:


df['_RFHLTH'].value_counts()


# ### Label feature
# - Keep only 1.0 values
# - Make 2.0 as 0.0 
# - Discard all other values
# - Rename the feature as `label`

# In[8]:


df['_RFHLTH'] = df['_RFHLTH'].replace({2: 0})
df = df.loc[df['_RFHLTH'].isin([0, 1])].copy()
df = df.rename(columns = {'_RFHLTH': 'label'})
df['label'].value_counts()


# ### What do you see?

# In[99]:


#We've successfully changed all values to binary


# Some housekeeping to make things smooth...

# In[9]:


# Remove columns with missing values
df = df.drop(columns = ['POORHLTH', 'PHYSHLTH', 'GENHLTH', 'PAINACT2', 
                        'QLMENTL2', 'QLSTRES2', 'QLHLTH2', 'HLTHPLN1', 'MENTHLTH'])


# ## Split Data into Training and Testing Set
# 
# Save 30% for testing

# In[10]:


from sklearn.model_selection import train_test_split


labels = np.array(df.pop('label'))


train, test, train_labels, test_labels = train_test_split(df, labels, 
                                                          stratify = labels,
                                                          test_size = 0.3, 
                                                          random_state = RSEED)


# #### Imputation of Missing values
# 
# We'll fill in the missing values with the mean of the column. It's important to note that we fill in missing values in the test set with the mean of columns in the training data. This is necessary because if we get new data, we'll have to use the training data to fill in any missing values. 

# In[11]:


train = train.fillna(train.mean())
test = test.fillna(train.mean())

# Features for feature importances, we will use this later below in this notebook
features = list(train.columns)


# In[12]:


train.shape


# In[13]:


test.shape


# ### Task Requirements
# - Build a RandomForest for the above dataset (not one but many with different sets of parameters)
# - Explore RandomizedSearchCV in Scikit-learn documentation
# - Create a parameter grid with these values
#     - n_estimators : between 10 and 200
#     - max_depth : choose between 3 and 20
#     - max_features : ['auto', 'sqrt', None] + list(np.arange(0.5, 1, 0.1))
#     - max_leaf_nodes : choose between 10 to 50
#     - min_samples_split : choose between 2, 5, or 10
#     - bootstrap : choose between True or False
# - Create the estimator (RandomForestClassifier)
# - Create the RandomizedSearchCV with estimator, parameter grid, scoring on roc auc, n_iter = 10, random_state=RSEED(50) for same reproducible results
# - Fit the model
# - Explore the best model parameters
# - Use the best model parameters to predict
# - Plot the best model ROC AUC Curve
# - Plot the Confusion Matrix
# - Write any insights or observations you found in the last

# ### Import RandomizedSearchCV

# In[14]:


from sklearn.model_selection import RandomizedSearchCV as CV


# ### Import RandomForestClassifier

# In[15]:


from sklearn.ensemble import RandomForestClassifier
from numpy import mean
from numpy import std
from numpy import arange
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from pprint import pprint


# ### Set the parameter grid according to the requirements above as a dictionary

# In[16]:


# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 200, num = 10)]

# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(3, 20, num = 10)]
max_depth.append(None)

# Max Number of features 
max_features = ['auto', 'sqrt', None] + list(np.arange(0.5, 1, 0.1))

#Setting the maximum amount of leaf nodes
max_leaf_nodes = [int(x) for x in np.linspace(10, 50, num = 10)]

# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]

# Method of selecting samples for training each tree
bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators,
               'max_depth': max_depth,
               'max_features': max_features,
               'max_leaf_nodes': max_leaf_nodes,
               'min_samples_split': min_samples_split,
               'bootstrap': bootstrap}


# ### Create the estimator with RSEED

# In[17]:


rfc = RandomForestClassifier()


# ### Create the Random Search model with cv=3, n_iter=10, scoring='roc_auc', random_state='RSEED'

# In[18]:


rfc_random = RandomizedSearchCV(estimator = rfc, param_distributions = random_grid, n_iter = 10, cv = 3, verbose=2, random_state=RSEED, n_jobs = -1)


# ### Fit the model 
# Note: It will take long time (around 20 - 1 hour depending on your computer specs). Good time to reload yourself with some energy or take a quick beauty nap!!!

# In[19]:


rfc_random.fit(train, train_labels)


# ### Explore the best parameters

# In[20]:


rfc_random.best_params_


# - First thing you'll notice is that the hyperparameter values are **not default** values.
# - Awesome. You've **tuned the hyperparameters**. Well done!!!

# ### Use the Best Model
# 
# Choose the best model as you find in under `best_estimator_`

# In[21]:


best_random = rfc_random.best_estimator_


# ### Make the predictions with the chosen best model

# In[22]:


rf_predictions = best_random.predict(test)
rf_probs = best_random.predict_proba(test)[:, 1]
tf_pred_train = best_random.predict(train)
tf_prob_train = best_random.predict_proba(train)[:, 1]


# ### Get the node counts and maximum depth of the random forest

# In[23]:


n_nodes = []
max_depths = []

# Stats about the trees in random forest
for ind_tree in best_random.estimators_:
    n_nodes.append(ind_tree.tree_.node_count)
    max_depths.append(ind_tree.tree_.max_depth)
    
print(f'Number of nodes {int(np.mean(n_nodes))}')
print(f'Maximum depth {int(np.mean(max_depths))}')


# ## Plot the ROC AUC Scores for training and testing data

# In[24]:


from sklearn.metrics import roc_auc_score
roc_value = roc_auc_score(test_labels, rf_probs)


# ### Helper function to Evaluate model

# In[25]:


from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_curve, auc

def evaluate_model(predictions, probs, train_predictions, train_probs):
    """Compare machine learning model to baseline performance.
    Computes statistics and shows ROC curve."""
    
    baseline = {}
    
    baseline['recall'] = recall_score(test_labels, [1 for _ in range(len(test_labels))])
    baseline['precision'] = precision_score(test_labels, [1 for _ in range(len(test_labels))])
    baseline['roc'] = 0.5
    
    results = {}
    
    results['recall'] = recall_score(test_labels, predictions)
    results['precision'] = precision_score(test_labels, predictions)
    results['roc'] = roc_auc_score(test_labels, probs)
    
    train_results = {}
    train_results['recall'] = recall_score(train_labels, train_predictions)
    train_results['precision'] = precision_score(train_labels, train_predictions)
    train_results['roc'] = roc_auc_score(train_labels, train_probs)
    
    for metric in ['recall', 'precision', 'roc']:
        print(f'{metric.capitalize()} Baseline: {round(baseline[metric], 2)} Test: {round(results[metric], 2)} Train: {round(train_results[metric], 2)}')
    
    # Calculate false positive rates and true positive rates
    base_fpr, base_tpr, _ = roc_curve(test_labels, [1 for _ in range(len(test_labels))])
    model_fpr, model_tpr, _ = roc_curve(test_labels, probs)

    plt.figure(figsize = (8, 6))
    plt.rcParams['font.size'] = 16
    
    # Plot both curves
    plt.plot(base_fpr, base_tpr, 'b', label = 'baseline')
    plt.plot(model_fpr, model_tpr, 'r', label = 'model')
    plt.legend();
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title('ROC Curves');


# ### Evaluate the best model
# - Plot the ROC AUC Curve

# In[26]:


evaluate_model(rf_predictions, rf_probs, tf_pred_train, tf_prob_train) 


# ### Confusion Matrix Helper function

# In[27]:


from sklearn.metrics import confusion_matrix
import itertools

#  Helper function to plot Confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Oranges):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.figure(figsize = (10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, size = 24)
    plt.colorbar(aspect=4)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, size = 14)
    plt.yticks(tick_marks, classes, size = 14)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    
    # Labeling the plot
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), fontsize = 20,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        
    plt.grid(None)
    plt.tight_layout()
    plt.ylabel('True label', size = 18)
    plt.xlabel('Predicted label', size = 18)


# # Please do not run the below 2 cells....
# ## It is given only for your comparision of Decision Tree, RandomForest and your very own Best RandomForest

# In[ ]:


# Decision Tree Confusion Matrix


# In[ ]:


# Random Forest


# ### Evaluate the best model
# - Plot Confusion Matrix

# In[ ]:


cm = confusion_matrix(test_labels, rf_predictions)
np.set_printoptions(precision=2)

plot_confusion_matrix(cm, labels)


# ### Observations / Insights ???

# In[ ]:


#It appears that most of the values appear to be True Negatives, and it is an imbalanced dataset.  It also appears we have more false negatives than True positives or false positives.


# ### Bonus: What if you want to explain your best RandomForest to your boss on the way it split the features??? Do not fret. Capture the estimator and convert them into a .png and present it in the meeting and get accolodes.

# In[ ]:


from sklearn.tree import export_graphviz
from subprocess import call
from IPython.display import Image

estimator = best_model.estimators_[1]

# Export a tree from the forest
export_graphviz(estimator, 'tree_from_optimized_forest.dot', rounded = True, 
                feature_names=train.columns, max_depth = 8, 
                class_names = ['poverty', 'no poverty'], filled = True)


# In[ ]:


call(['dot', '-Tpng', 'tree_from_optimized_forest.dot', '-o', 'tree_from_optimized_forest.png', '-Gdpi=200'])
Image('tree_from_optimized_forest.png')


# In[ ]:





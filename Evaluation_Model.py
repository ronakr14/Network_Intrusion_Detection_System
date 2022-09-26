#!/usr/bin/env python
# coding: utf-8

# # Import & Configs

# In[1]:


import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import matplotlib.pyplot as plt
import pickle

from sklearn.metrics import accuracy_score, precision_score, recall_score


# # No Feature Selection 

# ## Unpickling the data from preprocess

# In[12]:


X_train = pickle.load(open("C:/Users/Rathore/Downloads/NIDS/SKlearn/Pickle_Files/X_train.pkl",'rb'))
X_test = pickle.load(open("C:/Users/Rathore/Downloads/NIDS/SKlearn/Pickle_Files/X_test.pkl",'rb'))
X_train_dtc_df = pickle.load(open("C:/Users/Rathore/Downloads/NIDS/SKlearn/Pickle_Files/X_train_dtc_df.pkl",'rb'))
X_test_dtc_df = pickle.load(open("C:/Users/Rathore/Downloads/NIDS/SKlearn/Pickle_Files/X_test_dtc_df.pkl",'rb'))
X_train_rfe_dtc_df = pickle.load(open("C:/Users/Rathore/Downloads/NIDS/SKlearn/Pickle_Files/X_train_rfe_dtc_df.pkl",'rb'))
X_test_rfe_dtc_df = pickle.load(open("C:/Users/Rathore/Downloads/NIDS/SKlearn/Pickle_Files/X_test_rfe_dtc_df.pkl",'rb'))
Y_train = pickle.load(open("C:/Users/Rathore/Downloads/NIDS/SKlearn/Pickle_Files/Y_train.pkl",'rb'))
Y_test = pickle.load(open("C:/Users/Rathore/Downloads/NIDS/SKlearn/Pickle_Files/Y_test.pkl",'rb'))


# ## Unpickling the Models

# In[3]:


models={}
models['Bernoulli_NFS']=pickle.load(open("C:/Users/Rathore/Downloads/NIDS/SKlearn/Pickle_Files/Bernoulli_NFS.pkl",'rb'))
models['Decision_Tree_Classifier_NFS']=pickle.load(open("C:/Users/Rathore/Downloads/NIDS/SKlearn/Pickle_Files/Decision_Tree_Classifier_NFS.pkl",'rb'))
models['Guassian_NFS']=pickle.load(open("C:/Users/Rathore/Downloads/NIDS/SKlearn/Pickle_Files/Guassian_NFS.pkl",'rb'))
models['K_Nearest_Neighbours_NFS']=pickle.load(open("C:/Users/Rathore/Downloads/NIDS/SKlearn/Pickle_Files/K_Nearest_Neighbours_NFS.pkl",'rb'))
models['Logistic_Regression_NFS']=pickle.load(open("C:/Users/Rathore/Downloads/NIDS/SKlearn/Pickle_Files/Logistic_Regression_NFS.pkl",'rb'))
models['Random_Forest_Classifier_NFS']=pickle.load(open("C:/Users/Rathore/Downloads/NIDS/SKlearn/Pickle_Files/Random_Forest_Classifier_NFS.pkl",'rb'))
models['XGBoost_NFS']=pickle.load(open("C:/Users/Rathore/Downloads/NIDS/SKlearn/Pickle_Files/XGBoost_NFS.pkl",'rb'))


# ## Accuracy, Precision and Recall Scores

# In[4]:


accuracy, precision, recall = {}, {}, {}

for key in models.keys():
    Y_pred = models[key].predict(X_test)
    accuracy[key] = accuracy_score(Y_pred, Y_test)*100
    precision[key] = precision_score(Y_pred, Y_test)*100
    recall[key] = recall_score(Y_pred, Y_test)*100


# In[5]:


df_model = pd.DataFrame(index=models.keys(), columns=['Accuracy', 'Precision', 'Recall'])
df_model['Accuracy'] = accuracy.values()
df_model['Precision'] = precision.values()
df_model['Recall'] = recall.values()
df_model


# In[28]:


plt.rcParams["figure.figsize"] = (10,7)
plt.rcParams['axes.labelsize'] = 7
plt.rcParams['xtick.labelsize'] = 7
plt.rcParams['ytick.labelsize'] = 10


# In[29]:


ax = df_model.plot.barh()
ax.legend(
    ncol=len(models.keys()), 
    bbox_to_anchor=(0, 1), 
    loc='lower left', 
    prop={'size': 12}
)

plt.tight_layout()


# # Feature Selection with Decision Tree

# ## Unpickling the Models

# In[13]:


models={}
models['Bernoulli_FS_DTC']=pickle.load(open("C:/Users/Rathore/Downloads/NIDS/SKlearn/Pickle_Files/Bernoulli_FS_DTC.pkl",'rb'))
models['Decision_Tree_Classifier_FS_DTC']=pickle.load(open("C:/Users/Rathore/Downloads/NIDS/SKlearn/Pickle_Files/Decision_Tree_Classifier_FS_DTC.pkl",'rb'))
models['Gaussian_FS_DTC']=pickle.load(open("C:/Users/Rathore/Downloads/NIDS/SKlearn/Pickle_Files/Gaussian_FS_DTC.pkl",'rb'))
models['K_Nearest_Neighbor_FS_DTC']=pickle.load(open("C:/Users/Rathore/Downloads/NIDS/SKlearn/Pickle_Files/K_Nearest_Neighbor_FS_DTC.pkl",'rb'))
models['Logistic_Regression_FS_DTC']=pickle.load(open("C:/Users/Rathore/Downloads/NIDS/SKlearn/Pickle_Files/Logistic_Regression_FS_DTC.pkl",'rb'))
models['Random_Forest_Classifier_FS_DTC']=pickle.load(open("C:/Users/Rathore/Downloads/NIDS/SKlearn/Pickle_Files/Random_Forest_Classifier_FS_DTC.pkl",'rb'))
models['XGBoost_FS_DTC']=pickle.load(open("C:/Users/Rathore/Downloads/NIDS/SKlearn/Pickle_Files/XGBoost_FS_DTC.pkl",'rb'))


# ## Accuracy, Precision and Recall Scores

# In[14]:


accuracy, precision, recall = {}, {}, {}

for key in models.keys():
    Y_pred = models[key].predict(X_test_dtc_df)
    accuracy[key] = accuracy_score(Y_pred, Y_test)*100
    precision[key] = precision_score(Y_pred, Y_test)*100
    recall[key] = recall_score(Y_pred, Y_test)*100


# In[15]:


df_model = pd.DataFrame(index=models.keys(), columns=['Accuracy', 'Precision', 'Recall'])
df_model['Accuracy'] = accuracy.values()
df_model['Precision'] = precision.values()
df_model['Recall'] = recall.values()
df_model


# In[30]:


ax = df_model.plot.barh()
ax.legend(
    ncol=len(models.keys()), 
    bbox_to_anchor=(0, 1), 
    loc='lower left', 
    prop={'size': 12}
)

plt.tight_layout()


# # Feature Selection with Recursive Feature Elimination-DTC

# ## Unpickling the Models

# In[17]:


models={}
models['Bernoulli_FS_RFE_DTC']=pickle.load(open("C:/Users/Rathore/Downloads/NIDS/SKlearn/Pickle_Files/Bernoulli_FS_RFE_DTC.pkl",'rb'))
models['Decision_Tree_Classifier_FS_RFE_DTC']=pickle.load(open("C:/Users/Rathore/Downloads/NIDS/SKlearn/Pickle_Files/Decision_Tree_Classifier_FS_RFE_DTC.pkl",'rb'))
models['Guassian_FS_RFE_DTC']=pickle.load(open("C:/Users/Rathore/Downloads/NIDS/SKlearn/Pickle_Files/Guassian_FS_RFE_DTC.pkl",'rb'))
models['K_Nearest_Neighbor_FS_RFE_DTC']=pickle.load(open("C:/Users/Rathore/Downloads/NIDS/SKlearn/Pickle_Files/K_Nearest_Neighbor_FS_RFE_DTC.pkl",'rb'))
models['Logistic_Regression_FS_RFE_DTC']=pickle.load(open("C:/Users/Rathore/Downloads/NIDS/SKlearn/Pickle_Files/Logistic_Regression_FS_RFE_DTC.pkl",'rb'))
models['Random_Forest_Classifier_FS_RFE_DTC']=pickle.load(open("C:/Users/Rathore/Downloads/NIDS/SKlearn/Pickle_Files/Random_Forest_Classifier_FS_RFE_DTC.pkl",'rb'))
models['XGBoost_FS_RFE_DTC']=pickle.load(open("C:/Users/Rathore/Downloads/NIDS/SKlearn/Pickle_Files/XGBoost_FS_RFE_DTC.pkl",'rb'))


# ## Accuracy, Precision and Recall Scores

# In[19]:


accuracy, precision, recall = {}, {}, {}

for key in models.keys():
    Y_pred = models[key].predict(X_test_rfe_dtc_df)
    accuracy[key] = accuracy_score(Y_pred, Y_test)*100
    precision[key] = precision_score(Y_pred, Y_test)*100
    recall[key] = recall_score(Y_pred, Y_test)*100


# In[20]:


df_model = pd.DataFrame(index=models.keys(), columns=['Accuracy', 'Precision', 'Recall'])
df_model['Accuracy'] = accuracy.values()
df_model['Precision'] = precision.values()
df_model['Recall'] = recall.values()
df_model


# In[31]:


ax = df_model.plot.barh()
ax.legend(
    ncol=len(models.keys()), 
    bbox_to_anchor=(0, 1), 
    loc='lower left', 
    prop={'size': 12}
)

plt.tight_layout()


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# # Import and configs

# In[1]:


import warnings
warnings.filterwarnings('ignore')

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA


# # Unpickling files from preprocessing file

# In[2]:


X_train = pickle.load(open("C:/Users/Rathore/Downloads/NIDS/SKlearn/Pickle_Files/X_train.pkl",'rb'))
Y_train = pickle.load(open('C:/Users/Rathore/Downloads/NIDS/SKlearn/Pickle_Files/Y_train.pkl','rb'))
X_test = pickle.load(open("C:/Users/Rathore/Downloads/NIDS/SKlearn/Pickle_Files/X_test.pkl",'rb'))


# # Feature Selection

# ## Decision Tree Classifier

# In[3]:


from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(random_state=7);
dtc.fit(X_train, Y_train);
score = np.round(dtc.feature_importances_,3)
importances = pd.DataFrame({'feature':X_train.columns,'importance':score})
importances = importances.sort_values('importance',ascending=False).set_index('feature')


# In[4]:


plt.rcParams["figure.figsize"] = (17,5)
plt.rcParams['axes.labelsize'] = 7
plt.rcParams['xtick.labelsize'] = 7
plt.rcParams['ytick.labelsize'] = 10


# In[5]:


importances.plot.bar();


# In[6]:


df=importances[importances['importance']>0].index
dtc_sample_col=[x for x in df]
print(dtc_sample_col,len(dtc_sample_col))


# ### Pickling Train and Test Dataframe after DTC

# In[7]:


X_train_dtc_df=X_train[dtc_sample_col]
X_test_dtc_df=X_test[dtc_sample_col]
pickle.dump(X_train_dtc_df, open('X_train_dtc_df.pkl','wb'))
pickle.dump(X_test_dtc_df, open('X_test_dtc_df.pkl','wb'))


# ## Recursive Feature Selection using Decision Tree Classifier

# In[8]:


from sklearn.feature_selection import RFE
rfe_dt = RFE(DecisionTreeClassifier(random_state=7))
rfe_dt.fit(X_train,Y_train)
for col,rank in zip(X_train.columns,rfe_dt.ranking_):
    print(col, rank)
rfe_dtc_sample_col = X_train.columns[rfe_dt.ranking_ == 1]
print(rfe_dtc_sample_col,len(rfe_dtc_sample_col))


# ### Pickling Train and Test Dataframe after RFE-DTC

# In[9]:


X_train_rfe_dtc_df=X_train[rfe_dtc_sample_col]
X_test_rfe_dtc_df=X_test[rfe_dtc_sample_col]
pickle.dump(X_train_rfe_dtc_df, open('X_train_rfe_dtc_df.pkl','wb'))
pickle.dump(X_test_rfe_dtc_df, open('X_test_rfe_dtc_df.pkl','wb'))


# ## Principal Component Analysis

# In[10]:


from sklearn.decomposition import PCA
pca_9999 = PCA( n_components= 0.9999, random_state=7)
pca_9999.fit(X_train)
print(pca_9999.explained_variance_ratio_)
print(len(pca_9999.components_))
plt.plot(pca_9999.explained_variance_ratio_)


# ### Pickling Train and Test Dataframe after PCA

# In[11]:


train_pca_9999_df=pca_9999.transform(X_train)
test_pca_9999_df=pca_9999.transform(X_test)
pickle.dump(train_pca_9999_df, open('train_pca_9999_df.pkl','wb'))
pickle.dump(test_pca_9999_df, open('test_pca_9999_df.pkl','wb'))


# In[ ]:





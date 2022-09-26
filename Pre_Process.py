#!/usr/bin/env python
# coding: utf-8

# # Imports & Configs

# In[1]:


import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import seaborn as sns
import pickle

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split


# # Data Preprocessing

# In[2]:


df=pd.read_csv("C:/Users/Rathore/Downloads/NIDS/Datasets/NF-UNSW-NB15-v2.csv")
print("Dataset has {} rows & {} columns".format(df.shape[0],df.shape[1]))


# In[3]:


df.columns


# In[4]:


df.info()


# In[5]:


for col in df.columns:
    print('Column: ',col)
    print('Unique Values: ',df[col].unique())
    print()


# In[6]:


X=df.drop(['Label', 'DNS_QUERY_ID', 'Attack'],axis=1)
Y=df['Label']
print(X.shape,Y.shape)


# # Label Encoding

# In[7]:


cat_columns=['IPV4_SRC_ADDR', 'L4_SRC_PORT', 'IPV4_DST_ADDR', 'L4_DST_PORT','PROTOCOL', 'L7_PROTO','TCP_FLAGS', 'CLIENT_TCP_FLAGS', 'SERVER_TCP_FLAGS','ICMP_TYPE', 'ICMP_IPV4_TYPE', 'DNS_QUERY_TYPE','DNS_TTL_ANSWER', 'FTP_COMMAND_RET_CODE']
X_cat=X[cat_columns]
print(X_cat.shape)
X_cat.head().transpose()


# In[8]:


X_cat_OHE = pd.get_dummies(X_cat)
X_cat_OHE.shape


# In[9]:


num_col=[]
for col in X.columns:
    if col not in cat_columns:
        num_col.append(col)
print(num_col)


# In[10]:


X_num=X[num_col]
X_num.shape


# # Imputing the Outliers

# In[11]:


def min_max_iqr(df,col):
    q1, q3 = df[col].quantile([0.25,0.75])
    IQR = q3-q1
    min_valid = q1 - 1.5*IQR
    max_valid = q3 + 1.5 * IQR
    return min_valid,max_valid

for col in X_num.columns:
    min_valid,max_valid = min_max_iqr(X_num,col)
    outlier_rows = X_num[(X_num[col] < min_valid) | (X_num[col] > max_valid)]
    print("Before Outlier Management")
    print(col,':', str((outlier_rows.shape[0]/X_num.shape[0])*100)+'%')
    X_num.loc[ X_num[col] < min_valid , col] = min_valid
    X_num.loc[ X_num[col] > max_valid , col] = max_valid
    outlier_rows1 = X_num[(X_num[col] < min_valid) | (X_num[col] > max_valid)]
    print("After Outlier Management")
    print(col,':', str((outlier_rows1.shape[0]/X_num.shape[0])*100)+'%')
    print()


# # Standard Scalar

# In[12]:


scaler = StandardScaler()
X_num_sc = scaler.fit_transform(X_num)
X_num_sc_df = pd.DataFrame(X_num, columns = X_num.columns)


# In[13]:


X_1 = pd.concat([X_num_sc_df, X_cat_OHE],axis=1)
X_1.shape


# # Split the Data

# In[14]:


X_train, X_test, Y_train, Y_test = train_test_split(X_1,Y, test_size=0.3, random_state= 7)
print(X_train.shape,X_test.shape,Y_train.shape,Y_test.shape)


# In[15]:


Y_train.value_counts()


# # Undersampling 

# In[16]:


X_train0 = X_train.loc[ Y_train ==0 ,:] 
X_train1 = X_train.loc[ Y_train ==1 ,:] 
print(X_train0.shape,X_train1.shape)

X_train0 = X_train.loc[ Y_train ==0 ,:] 
X_train1 = X_train.loc[ Y_train ==1 ,:] 
X_train0=X_train0.sample(X_train1.shape[0],random_state=7)
X_train_un=pd.concat([X_train0,X_train1],axis=0)
Y_train_un=Y_train.loc[X_train_un.index]

X_train_un.shape,Y_train_un.shape


# In[17]:


Y_train_un.value_counts()


# # Pickling the undersample data

# In[18]:


pickle.dump(X_train_un, open('X_train.pkl','wb'))
pickle.dump(Y_train_un, open('Y_train.pkl','wb'))
pickle.dump(X_test, open('X_test.pkl','wb'))
pickle.dump(Y_test, open('Y_test.pkl','wb'))


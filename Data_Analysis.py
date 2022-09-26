#!/usr/bin/env python
# coding: utf-8

# # Imports & Configs

# In[1]:


import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# Settings
pd.set_option('display.max_columns', None)
np.set_printoptions(precision=3)
sns.set(style="darkgrid")
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams["figure.figsize"] = (5,5)


# # Exploratory Data Analysis

# In[3]:


df=pd.read_csv("C:/Users/Rathore/Downloads/NIDS/Datasets/NF-UNSW-NB15-v2.csv")
print("Testing data has {} rows & {} columns".format(df.shape[0],df.shape[1]))


# In[4]:


df.head()


# In[5]:


df.tail()


# In[6]:


df.head().transpose()


# In[7]:


df.info()


# In[8]:


df.describe().transpose()


# In[9]:


df.describe(include = 'all').transpose()


# In[10]:


df.duplicated().sum()


# In[11]:


df.isnull().sum()


# In[12]:


for col in df.columns:
    print("----------",col,"----------")
    print(df[col].value_counts())
    print()


# In[13]:


sns.countplot(df['Label'])


# In[106]:


plt.rcParams["figure.figsize"] = (10,10)
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10


# In[108]:


ax = df.groupby(['Attack'])['Attack'].count().plot(kind='barh')
plt.tight_layout()


# In[16]:


df.groupby(['Attack'])['Attack'].count()


# In[17]:


plt.rcParams["figure.figsize"] = (15,5)
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10


# In[18]:


sns.boxplot(x = 'L4_SRC_PORT', data = df)


# In[19]:


sns.boxplot(x = 'L4_DST_PORT', data = df)


# In[20]:


sns.boxplot(x = 'PROTOCOL', data = df)


# In[21]:


sns.boxplot(x = 'L7_PROTO', data = df)


# In[22]:


sns.boxplot(x = 'IN_BYTES', data = df)


# In[23]:


sns.boxplot(x = 'OUT_BYTES', data = df)


# In[24]:


sns.boxplot(x = 'IN_PKTS', data = df)


# In[25]:


sns.boxplot(x = 'TCP_FLAGS', data = df)


# In[26]:


sns.boxplot(x = 'CLIENT_TCP_FLAGS', data = df)


# In[27]:


sns.boxplot(x = 'SERVER_TCP_FLAGS', data = df)


# In[28]:


sns.boxplot(x = 'FLOW_DURATION_MILLISECONDS', data = df)


# In[29]:


sns.boxplot(x = 'DURATION_IN', data = df)


# In[30]:


sns.boxplot(x = 'DURATION_OUT', data = df)


# In[31]:


sns.boxplot(x = 'MIN_TTL', data = df)


# In[32]:


sns.boxplot(x = 'MAX_TTL', data = df)


# In[33]:


sns.boxplot(x = 'LONGEST_FLOW_PKT', data = df)


# In[34]:


sns.boxplot(x = 'SHORTEST_FLOW_PKT', data = df)


# In[35]:


sns.boxplot(x = 'MIN_IP_PKT_LEN', data = df)


# In[36]:


sns.boxplot(x = 'MAX_IP_PKT_LEN', data = df)


# In[37]:


sns.boxplot(x = 'SRC_TO_DST_SECOND_BYTES', data = df)


# In[38]:


sns.boxplot(x = 'DST_TO_SRC_SECOND_BYTES', data = df)


# In[39]:


sns.boxplot(x = 'RETRANSMITTED_IN_BYTES', data = df)


# In[40]:


sns.boxplot(x = 'RETRANSMITTED_IN_PKTS', data = df)


# In[41]:


sns.boxplot(x = 'RETRANSMITTED_OUT_PKTS', data = df)


# In[42]:


sns.boxplot(x = 'SRC_TO_DST_AVG_THROUGHPUT', data = df)


# In[43]:


sns.boxplot(x = 'DST_TO_SRC_AVG_THROUGHPUT', data = df)


# In[44]:


sns.boxplot(x = 'NUM_PKTS_UP_TO_128_BYTES', data = df)


# In[45]:


sns.boxplot(x = 'NUM_PKTS_128_TO_256_BYTES', data = df)


# In[46]:


sns.boxplot(x = 'NUM_PKTS_256_TO_512_BYTES', data = df)


# In[47]:


sns.boxplot(x = 'NUM_PKTS_512_TO_1024_BYTES', data = df)


# In[48]:


sns.boxplot(x = 'NUM_PKTS_1024_TO_1514_BYTES', data = df)


# In[49]:


sns.boxplot(x = 'TCP_WIN_MAX_IN', data = df)


# In[50]:


sns.boxplot(x = 'TCP_WIN_MAX_OUT', data = df)


# In[51]:


sns.boxplot(x = 'ICMP_TYPE', data = df)


# In[52]:


sns.boxplot(x = 'ICMP_IPV4_TYPE', data = df)


# In[53]:


sns.boxplot(x = 'DNS_QUERY_ID', data = df)


# In[54]:


sns.boxplot(x = 'DNS_QUERY_TYPE', data = df)


# In[55]:


sns.boxplot(x = 'DNS_TTL_ANSWER', data = df)


# In[56]:


sns.boxplot(x = 'FTP_COMMAND_RET_CODE', data = df)


# In[57]:


plt.rcParams["figure.figsize"] = (5,5)
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10


# In[58]:


df.L4_SRC_PORT.plot(kind='hist' , grid = True,xlabel="L4_SRC_PORT")


# In[59]:


df.L4_DST_PORT.plot(kind='hist' , grid = True)


# In[60]:


df.PROTOCOL.plot(kind='hist' , grid = True)


# In[61]:


df.L7_PROTO.plot(kind='hist' , grid = True)


# In[62]:


df.IN_BYTES.plot(kind='hist' , grid = True)


# In[63]:


df.OUT_BYTES.plot(kind='hist' , grid = True)


# In[64]:


df.IN_PKTS.plot(kind='hist' , grid = True)


# In[65]:


df.OUT_PKTS.plot(kind='hist' , grid = True)


# In[66]:


df.TCP_FLAGS.plot(kind='hist' , grid = True)


# In[67]:


df.OUT_PKTS.plot(kind='hist' , grid = True)


# In[68]:


df.CLIENT_TCP_FLAGS.plot(kind='hist' , grid = True)


# In[69]:


df.SERVER_TCP_FLAGS.plot(kind='hist' , grid = True)


# In[70]:


df.FLOW_DURATION_MILLISECONDS.plot(kind='hist' , grid = True)


# In[71]:


df.DURATION_IN.plot(kind='hist' , grid = True)


# In[72]:


df.DURATION_OUT.plot(kind='hist' , grid = True)


# In[73]:


df.MIN_TTL.plot(kind='hist' , grid = True)


# In[74]:


df.MAX_TTL.plot(kind='hist' , grid = True)


# In[75]:


df.LONGEST_FLOW_PKT.plot(kind='hist' , grid = True)


# In[76]:


df.SHORTEST_FLOW_PKT.plot(kind='hist' , grid = True)


# In[77]:


df.MIN_IP_PKT_LEN.plot(kind='hist' , grid = True)


# In[78]:


df.MAX_IP_PKT_LEN.plot(kind='hist' , grid = True)


# In[79]:


df.SRC_TO_DST_SECOND_BYTES.plot(kind='hist' , grid = True)


# In[80]:


df.DST_TO_SRC_SECOND_BYTES.plot(kind='hist' , grid = True)


# In[81]:


df.RETRANSMITTED_IN_BYTES.plot(kind='hist' , grid = True)


# In[82]:


df.RETRANSMITTED_IN_PKTS.plot(kind='hist' , grid = True)


# In[83]:


df.RETRANSMITTED_OUT_BYTES.plot(kind='hist' , grid = True)


# In[84]:


df.RETRANSMITTED_OUT_PKTS.plot(kind='hist' , grid = True)


# In[85]:


df.SRC_TO_DST_AVG_THROUGHPUT.plot(kind='hist' , grid = True)


# In[86]:


df.DST_TO_SRC_AVG_THROUGHPUT.plot(kind='hist' , grid = True)


# In[87]:


df.NUM_PKTS_UP_TO_128_BYTES.plot(kind='hist' , grid = True)


# In[88]:


df.NUM_PKTS_128_TO_256_BYTES.plot(kind='hist' , grid = True)


# In[89]:


df.NUM_PKTS_256_TO_512_BYTES.plot(kind='hist' , grid = True)


# In[90]:


df.NUM_PKTS_512_TO_1024_BYTES.plot(kind='hist' , grid = True)


# In[91]:


df.NUM_PKTS_1024_TO_1514_BYTES.plot(kind='hist' , grid = True)


# In[92]:


df.TCP_WIN_MAX_IN.plot(kind='hist' , grid = True)


# In[93]:


df.TCP_WIN_MAX_OUT.plot(kind='hist' , grid = True)


# In[94]:


df.ICMP_TYPE.plot(kind='hist' , grid = True)


# In[95]:


df.ICMP_IPV4_TYPE.plot(kind='hist' , grid = True)


# In[96]:


df.DNS_QUERY_ID.plot(kind='hist' , grid = True)


# In[97]:


df.DNS_QUERY_TYPE.plot(kind='hist' , grid = True)


# In[98]:


df.DNS_TTL_ANSWER.plot(kind='hist' , grid = True)


# In[99]:


df.FTP_COMMAND_RET_CODE.plot(kind='hist' , grid = True)


# In[100]:


for col in df.select_dtypes(include = ['int64','float64']):
    print("----------",col,"----------")
    print(f"Skewness: {df[col].skew()}")
    print(f"Kurtosis: {df[col].kurt()}")
    print()


# In[101]:


plt.rcParams["figure.figsize"] = (200,200)
plt.rcParams['axes.labelsize'] = 100
plt.rcParams['xtick.labelsize'] = 100
plt.rcParams['ytick.labelsize'] = 100


# In[102]:


corrmat = df.corr()
hm = sns.heatmap(corrmat, 
                 cbar=True, 
                 annot=True, 
                 square=True, 
                 fmt='.2f', 
                 annot_kws={'size': 100}, 
                 yticklabels=df.columns, 
                 xticklabels=df.columns, 
                 cmap="Spectral_r")
plt.show()


# In[103]:


mask = np.zeros_like(corrmat)
mask[np.triu_indices_from(mask)] = True
hm = sns.heatmap(corrmat, 
                 cbar=True, 
                 annot=True, 
                 square=True,
                 mask=mask,
                 fmt='.2f', 
                 annot_kws={'size': 100}, 
                 yticklabels=df.columns, 
                 xticklabels=df.columns, 
                 cmap="Spectral_r")
plt.show()


# In[104]:


df.select_dtypes(['float64' , 'int64']).corr()


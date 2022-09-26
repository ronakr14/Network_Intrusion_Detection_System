#!/usr/bin/env python
# coding: utf-8

# # Import & configs

# In[1]:


import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import average_precision_score


# # Unpickling data from preprocess

# In[2]:


X_train = pickle.load(open("C:/Users/Rathore/Downloads/NIDS/SKlearn/Pickle_Files/X_train.pkl",'rb'))
X_test = pickle.load(open("C:/Users/Rathore/Downloads/NIDS/SKlearn/Pickle_Files/X_test.pkl",'rb'))
Y_train = pickle.load(open("C:/Users/Rathore/Downloads/NIDS/SKlearn/Pickle_Files/Y_train.pkl",'rb'))
Y_test = pickle.load(open("C:/Users/Rathore/Downloads/NIDS/SKlearn/Pickle_Files/Y_test.pkl",'rb'))


# # Logistic Regression Model

# In[3]:


logisticRegr = LogisticRegression()
logisticRegr.fit(X_train, Y_train)
Y_pred_lr=logisticRegr.predict(X_test)


# # Pickling the model 

# In[4]:


pickle.dump(logisticRegr, open('Logistic_Regression_NFS.pkl','wb'))


# # Accuracy Score

# In[5]:


accuracy = accuracy_score(Y_test, Y_pred_lr)
print("Accuracy Score:",accuracy*100)


# # Cross Validation Score

# In[6]:


accuracy_train = cross_val_score(logisticRegr, X_train, Y_train, cv=5)
print(accuracy_train)
print("Cross Validation Accuracy for train data is:",accuracy_train.mean() * 100)


# # Classification Report

# In[7]:


class_rep=classification_report(Y_test, Y_pred_lr)
print("----------Classification Report----------")
print(class_rep)


# # Confusion Matrix 

# In[8]:


ConfusionMatrixDisplay.from_predictions(Y_test, Y_pred_lr)


# In[9]:


cf_matrix=confusion_matrix(Y_test, Y_pred_lr)
sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
            fmt='.2%', cmap='PiYG')


# # ROC Curve

# In[10]:


RocCurveDisplay.from_predictions(Y_test, Y_pred_lr)


# # Precision Recall Curve

# In[11]:


PrecisionRecallDisplay.from_predictions(Y_test, Y_pred_lr)


# # Average Precision Score 

# In[12]:


avg_precision = average_precision_score(Y_test, Y_pred_lr)
print("Average Precision:",avg_precision)


# # Extra

# In[13]:


tn, fp, fn, tp = cf_matrix.ravel()
false_positive_rate = fp / (fp + tn)
print("False Positive Rate:",false_positive_rate)
false_negative_rate = fn / (tp + fn)
print("False_Negative_Rate:",false_negative_rate)
true_negative_rate = tn / (tn + fp)
print("True_Negative_Rate:",true_negative_rate)
negative_predictive_value = tn/ (tn + fn)
print("Negative_Predictive_Value:",negative_predictive_value)
false_discovery_rate = fp/ (tp + fp)
print("False_Discovery_Rate:",false_discovery_rate)


# In[ ]:





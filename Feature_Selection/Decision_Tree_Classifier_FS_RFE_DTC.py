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

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

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


X_train_rfe_dtc_df = pickle.load(open("C:/Users/Rathore/Downloads/NIDS/SKlearn/Pickle_Files/X_train_rfe_dtc_df.pkl",'rb'))
X_test_rfe_dtc_df = pickle.load(open("C:/Users/Rathore/Downloads/NIDS/SKlearn/Pickle_Files/X_test_rfe_dtc_df.pkl",'rb'))
Y_train = pickle.load(open("C:/Users/Rathore/Downloads/NIDS/SKlearn/Pickle_Files/Y_train.pkl",'rb'))
Y_test = pickle.load(open("C:/Users/Rathore/Downloads/NIDS/SKlearn/Pickle_Files/Y_test.pkl",'rb'))


# # Grid Search

# In[3]:


param_grid={"criterion":["gini", "entropy","log_loss"],
"max_depth":[ 3, 5,7],
"ccp_alpha":[0.1,0.3,0.5,0.7,0.9]}
gscv=GridSearchCV(estimator=DecisionTreeClassifier(random_state=7),param_grid=param_grid,cv=2,verbose=2)
gscv.fit(X_train_rfe_dtc_df, Y_train)
print(gscv.best_params_)


# # Decision Tree Classifier Model

# In[4]:


dtc=DecisionTreeClassifier(criterion='entropy',max_depth=5,ccp_alpha=0.1,random_state=7)
dtc.fit(X_train_rfe_dtc_df, Y_train)
Y_pred_dtc=dtc.predict(X_test_rfe_dtc_df)
print("Parameters :",dtc.get_params())


# # Pickling the model 

# In[5]:


pickle.dump(dtc, open('Decision_Tree_Classifier_FS_RFE_DTC.pkl','wb'))


# # Accuracy Score

# In[6]:


accuracy = accuracy_score(Y_test, Y_pred_dtc)
print("Accuracy Score:",accuracy*100)


# # Cross Validation Score

# In[7]:


accuracy_train = cross_val_score(dtc, X_train_rfe_dtc_df, Y_train, cv=5)
print(accuracy_train)
print("Cross Validation Accuracy for train data is:",accuracy_train.mean() * 100)


# # Classification Report

# In[8]:


class_rep=classification_report(Y_test, Y_pred_dtc)
print("----------Classification Report----------")
print(class_rep)


# # Confusion Matrix 

# In[9]:


ConfusionMatrixDisplay.from_predictions(Y_test, Y_pred_dtc)


# In[11]:


cf_matrix=confusion_matrix(Y_test, Y_pred_dtc)
sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
            fmt='.2%', cmap='PiYG')


# # ROC Curve

# In[12]:


RocCurveDisplay.from_predictions(Y_test, Y_pred_dtc)


# # Precision Recall Curve

# In[13]:


PrecisionRecallDisplay.from_predictions(Y_test, Y_pred_dtc)


# # Average Precision Score 

# In[14]:


avg_precision = average_precision_score(Y_test, Y_pred_dtc)
print("Average Precision:",avg_precision)


# # Extra

# In[15]:


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





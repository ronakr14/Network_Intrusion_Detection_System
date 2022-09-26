#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Intialization
import os
import sys

os.environ["SPARK_HOME"] = "/home/talentum/spark"
os.environ["PYLIB"] = os.environ["SPARK_HOME"] + "/python/lib"
# In below two lines, use /usr/bin/python2.7 if you want to use Python 2
os.environ["PYSPARK_PYTHON"] = "/usr/bin/python3.6" 
os.environ["PYSPARK_DRIVER_PYTHON"] = "/usr/bin/python3"
sys.path.insert(0, os.environ["PYLIB"] +"/py4j-0.10.7-src.zip")
sys.path.insert(0, os.environ["PYLIB"] +"/pyspark.zip")

# NOTE: Whichever package you want mention here.
# os.environ['PYSPARK_SUBMIT_ARGS'] = '--packages com.databricks:spark-xml_2.11:0.6.0 pyspark-shell' 
# os.environ['PYSPARK_SUBMIT_ARGS'] = '--packages org.apache.spark:spark-avro_2.11:2.4.0 pyspark-shell'
os.environ['PYSPARK_SUBMIT_ARGS'] = '--packages com.databricks:spark-xml_2.11:0.6.0,org.apache.spark:spark-avro_2.11:2.4.3 pyspark-shell'
# os.environ['PYSPARK_SUBMIT_ARGS'] = '--packages com.databricks:spark-xml_2.11:0.6.0,org.apache.spark:spark-avro_2.11:2.4.0 pyspark-shell'


# In[2]:


#Entrypoint 2.x
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("Spark SQL basic example").enableHiveSupport().getOrCreate()

# On yarn:
# spark = SparkSession.builder.appName("Spark SQL basic example").enableHiveSupport().master("yarn").getOrCreate()
# specify .master("yarn")

sc = spark.sparkContext


# In[3]:


# Load the CSV file
filepath="NF-UNSW-NB15-v2.csv"
network_df = spark.read.csv(path=filepath,header=True,inferSchema=True)


# In[4]:


print(type(network_df))


# In[5]:


network_df.printSchema()


# In[6]:


network_df.count()


# In[7]:


print("There are {} rows in the DataFrame.".format(network_df.count()))
print("There are {} columns in the DataFrame and their names are {}"      .format(len(network_df.columns), network_df.columns))


# In[12]:


df=network_df.drop("IPV4_SRC_ADDR","IPV4_DST_ADDR","L4_SRC_PORT","L4_DST_PORT","Attack")


# In[13]:


print("There are {} rows in the DataFrame.".format(df.count()))
print("There are {} columns in the DataFrame and their names are {}"      .format(len(df.columns), df.columns))


# In[14]:


import pandas as pd
pd.DataFrame(df.take(5), columns=df.columns).transpose()


# In[15]:


numeric_features = [t[0] for t in df.dtypes if t[1] == 'int']
df.select(numeric_features).describe().toPandas().transpose()


# In[16]:


from pyspark.ml.feature import VectorAssembler

df_assembler = VectorAssembler(inputCols=['PROTOCOL', 'L7_PROTO', 'IN_BYTES', 'IN_PKTS', 'OUT_BYTES', 'OUT_PKTS', 'TCP_FLAGS', 'CLIENT_TCP_FLAGS', 'SERVER_TCP_FLAGS', 'FLOW_DURATION_MILLISECONDS', 'DURATION_IN', 'DURATION_OUT', 'MIN_TTL', 'MAX_TTL', 'LONGEST_FLOW_PKT', 'SHORTEST_FLOW_PKT', 'MIN_IP_PKT_LEN', 'MAX_IP_PKT_LEN', 'SRC_TO_DST_SECOND_BYTES', 'DST_TO_SRC_SECOND_BYTES', 'RETRANSMITTED_IN_BYTES', 'RETRANSMITTED_IN_PKTS', 'RETRANSMITTED_OUT_BYTES', 'RETRANSMITTED_OUT_PKTS', 'SRC_TO_DST_AVG_THROUGHPUT', 'DST_TO_SRC_AVG_THROUGHPUT', 'NUM_PKTS_UP_TO_128_BYTES', 'NUM_PKTS_128_TO_256_BYTES', 'NUM_PKTS_256_TO_512_BYTES', 'NUM_PKTS_512_TO_1024_BYTES', 'NUM_PKTS_1024_TO_1514_BYTES', 'TCP_WIN_MAX_IN', 'TCP_WIN_MAX_OUT', 'ICMP_TYPE', 'ICMP_IPV4_TYPE', 'DNS_QUERY_ID', 'DNS_QUERY_TYPE', 'DNS_TTL_ANSWER', 'FTP_COMMAND_RET_CODE'], outputCol="features")
df = df_assembler.transform(df)


# In[22]:


df_vector=df.select('features','Label')
df_vector.show(10)


# In[24]:


train_df,test_df=df_vector.randomSplit([0.75,0.25])


# In[27]:


from pyspark.ml.classification import LogisticRegression


# In[28]:


log_reg=LogisticRegression(labelCol='Label').fit(train_df)


# In[30]:


train_results=log_reg.evaluate(train_df).predictions
train_results.filter(train_results['Label']==1).filter(train_results['prediction']==1).select(['Label','prediction','probability']).show(10,False)


# In[31]:


import matplotlib.pyplot as plt
import numpy as np
beta = np.sort(log_reg.coefficients)
plt.plot(beta)
plt.ylabel('Beta Coefficients')
plt.show()


# In[32]:


trainingSummary = log_reg.summary
roc = trainingSummary.roc.toPandas()
plt.plot(roc['FPR'],roc['TPR'])
plt.ylabel('False Positive Rate')
plt.xlabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
print('Training set areaUnderROC: ' + str(trainingSummary.areaUnderROC))


# In[33]:


pr = trainingSummary.pr.toPandas()
plt.plot(pr['recall'],pr['precision'])
plt.ylabel('Precision')
plt.xlabel('Recall')
plt.show()


# In[37]:


predictions = log_reg.transform(test_df)
predictions.select('features', 'Label', 'rawPrediction', 'prediction', 'probability').show(10)


# In[40]:


from pyspark.ml.evaluation import BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator(labelCol= 'Label')
print('Test Area Under ROC', evaluator.evaluate(predictions))


# In[41]:


accuracy = predictions.filter(predictions.Label == predictions.prediction).count() / float(predictions.count())
print("Accuracy : ",accuracy)


# In[ ]:





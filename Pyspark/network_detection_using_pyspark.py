#!/usr/bin/env python
# coding: utf-8

# # Spark Intialization

# In[1]:


import os
import sys

os.environ["SPARK_HOME"] = "/home/talentum/spark"
os.environ["PYLIB"] = os.environ["SPARK_HOME"] + "/python/lib"
os.environ["PYSPARK_PYTHON"] = "/usr/bin/python3.6" 
os.environ["PYSPARK_DRIVER_PYTHON"] = "/usr/bin/python3"
sys.path.insert(0, os.environ["PYLIB"] +"/py4j-0.10.7-src.zip")
sys.path.insert(0, os.environ["PYLIB"] +"/pyspark.zip")

os.environ['PYSPARK_SUBMIT_ARGS'] = '--packages com.databricks:spark-xml_2.11:0.6.0,org.apache.spark:spark-avro_2.11:2.4.3 pyspark-shell'


# In[2]:


from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("ML-NIDS").enableHiveSupport().getOrCreate()

sc = spark.sparkContext


# # Reading Dataframe

# In[3]:


filepath="NF-UNSW-NB15-v2.csv"
network_df = spark.read.csv(path=filepath,header=True,inferSchema=True)
network_df.printSchema()


# In[4]:


print("There are {} rows in the DataFrame.".format(network_df.count()))
print("There are {} columns in the DataFrame and their names are {}".format(len(network_df.columns), network_df.columns))


# In[5]:


df=network_df.drop("IPV4_SRC_ADDR","IPV4_DST_ADDR","L4_SRC_PORT","L4_DST_PORT","Attack")


# # Vector Transformation

# In[6]:


from pyspark.ml.feature import VectorAssembler
df_assembler = VectorAssembler(inputCols=['PROTOCOL', 'L7_PROTO', 'IN_BYTES', 'IN_PKTS', 'OUT_BYTES', 'OUT_PKTS', 'TCP_FLAGS', 'CLIENT_TCP_FLAGS', 'SERVER_TCP_FLAGS', 'FLOW_DURATION_MILLISECONDS', 'DURATION_IN', 'DURATION_OUT', 'MIN_TTL', 'MAX_TTL', 'LONGEST_FLOW_PKT', 'SHORTEST_FLOW_PKT', 'MIN_IP_PKT_LEN', 'MAX_IP_PKT_LEN', 'SRC_TO_DST_SECOND_BYTES', 'DST_TO_SRC_SECOND_BYTES', 'RETRANSMITTED_IN_BYTES', 'RETRANSMITTED_IN_PKTS', 'RETRANSMITTED_OUT_BYTES', 'RETRANSMITTED_OUT_PKTS', 'SRC_TO_DST_AVG_THROUGHPUT', 'DST_TO_SRC_AVG_THROUGHPUT', 'NUM_PKTS_UP_TO_128_BYTES', 'NUM_PKTS_128_TO_256_BYTES', 'NUM_PKTS_256_TO_512_BYTES', 'NUM_PKTS_512_TO_1024_BYTES', 'NUM_PKTS_1024_TO_1514_BYTES', 'TCP_WIN_MAX_IN', 'TCP_WIN_MAX_OUT', 'ICMP_TYPE', 'ICMP_IPV4_TYPE', 'DNS_QUERY_ID', 'DNS_QUERY_TYPE', 'DNS_TTL_ANSWER', 'FTP_COMMAND_RET_CODE'], outputCol="features")
df = df_assembler.transform(df)
df_vector=df.select('features','Label')


# # Feature Selection

# In[8]:


from pyspark.ml.feature import VectorSlicer
slicer = VectorSlicer(inputCol="features", outputCol="slicedfeatures", indices=range(1,20,1))
df_fs = slicer.transform(df_vector)


# # Standard Scalar

# In[10]:


from pyspark.ml.feature import StandardScaler
Scalerizer=StandardScaler().setInputCol("slicedfeatures").setOutputCol("Scaled_features")
df_std=Scalerizer.fit(df_fs).transform(df_fs)


# In[11]:


df_std=df_std.select('Scaled_features','Label')
df_final = df_std.withColumnRenamed("Scaled_features","features").withColumnRenamed("Label","label")


# # Train-Test Split

# In[14]:


train_df,test_df=df_final.randomSplit([0.70,0.30])


# # Logistic Regression Model

# In[17]:


from pyspark.ml.classification import LogisticRegression
log_reg=LogisticRegression(labelCol='label')
logmodel=log_reg.fit(train_df)
pred_log = logmodel.transform(test_df)


# # Decision Tree Classifier

# In[18]:


from pyspark.ml.classification import DecisionTreeClassifier
dtc=DecisionTreeClassifier(labelCol='label')
dtcmodel=dtc.fit(train_df)
pred_dtc = dtcmodel.transform(test_df)


# # Naive Bayes 

# In[20]:


from pyspark.ml.classification import NaiveBayes
nb = NaiveBayes(modelType='multinomial')
nbmodel = nb.fit(train_df)
pred_nb = nbmodel.transform(test_df)


# # Random Forest

# In[21]:


from pyspark.ml.classification import RandomForestClassifier
rf = RandomForestClassifier(featuresCol = 'features', labelCol = 'label')
rfModel = rf.fit(train_df)
pred_rf = rfModel.transform(test_df)


# # GBTClassifier

# In[23]:


from pyspark.ml.classification import GBTClassifier
gbtc = GBTClassifier(labelCol="label", maxIter=10)
gbtc = gbtc.fit(train_df)
pred_gbt = gbtc.transform(test_df)


# # Accuracy Score

# In[49]:


models={}
models['Logistic_Regression']=logmodel
models['Decision_Tree_Classifier']=dtcmodel
models['Naive_Bayes_Multinomial']=nbmodel
models['Random_Forest']=rfModel
models['Gradient_Boost']=gbtc


# In[50]:


from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

eval_bin = BinaryClassificationEvaluator(labelCol= 'label')
eval_mul=MulticlassClassificationEvaluator(metricName="accuracy")


accuracy,AUC,test_accuracy = {},{},{}
for key in models.keys():
    pred = models[key].transform(test_df)
    accuracy[key] = pred.filter(pred.label == pred.prediction).count() / float(pred.count())
    AUC[key]=eval_bin.evaluate(pred)
    test_accuracy[key]=eval_mul.evaluate(pred)


# In[51]:


import pandas as pd
df_model = pd.DataFrame(index=models.keys(), columns=['Accuracy', 'AUC_Score', 'Test_Accuracy'])
df_model['Accuracy'] = accuracy.values()
df_model['AUC_Score'] = AUC.values()
df_model['Test_Accuracy'] = test_accuracy.values()
df_model


# In[52]:


import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (10,7)
plt.rcParams['axes.labelsize'] = 7
plt.rcParams['xtick.labelsize'] = 7
plt.rcParams['ytick.labelsize'] = 10


# In[53]:


ax = df_model.plot.barh()
ax.legend(
 ncol=len(models.keys()),
 bbox_to_anchor=(0, 1),
 loc='lower left',
 prop={'size': 12}
)
plt.tight_layout()


# # Beta Coefficient of Logistic Regression

# In[54]:


plt.rcParams["figure.figsize"] = (10,5)
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10


# In[55]:


import numpy as np
beta = np.sort(logmodel.coefficients)
plt.plot(beta)
plt.ylabel('Beta Coefficients')
plt.show()


# # ROC Curves

# In[56]:


from pyspark.mllib.evaluation import BinaryClassificationMetrics

class CurveMetrics(BinaryClassificationMetrics):
    def __init__(self, *args):
        super(CurveMetrics, self).__init__(*args)

    def _to_list(self, rdd):
        points = []
        
        for row in rdd.collect():
            points += [(float(row._1()), float(row._2()))]
        return points

    def get_curve(self, method):
        rdd = getattr(self._java_model, method)().toJavaRDD()
        return self._to_list(rdd)


# # Logistic Regression

# In[57]:


preds = pred_log.select('label','probability').rdd.map(lambda row: (float(row['probability'][1]), float(row['label'])))
points = CurveMetrics(preds).get_curve('roc')


# In[58]:


plt.rcParams["figure.figsize"] = (10,5)
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10


# In[59]:


plt.figure()
x_val = [x[0] for x in points]
y_val = [x[1] for x in points]
plt.title('Logistic Regression - ROC Curve')
plt.plot(x_val, y_val)


# In[60]:


preds = pred_dtc.select('label','probability').rdd.map(lambda row: (float(row['probability'][1]), float(row['label'])))
points = CurveMetrics(preds).get_curve('roc')


# In[61]:


plt.figure()
x_val = [x[0] for x in points]
y_val = [x[1] for x in points]
plt.title('Decision Tree Classifier - ROC Curve')
plt.plot(x_val, y_val)


# In[62]:


preds = pred_nb.select('label','probability').rdd.map(lambda row: (float(row['probability'][1]), float(row['label'])))
points = CurveMetrics(preds).get_curve('roc')


# In[63]:


plt.figure()
x_val = [x[0] for x in points]
y_val = [x[1] for x in points]
plt.title('Naive Bayes - ROC Curve')
plt.plot(x_val, y_val)


# In[64]:


preds = pred_rf.select('label','probability').rdd.map(lambda row: (float(row['probability'][1]), float(row['label'])))
points = CurveMetrics(preds).get_curve('roc')


# In[65]:


plt.figure()
x_val = [x[0] for x in points]
y_val = [x[1] for x in points]
plt.title('Random Forest Classifier - ROC Curve')
plt.plot(x_val, y_val)


# In[66]:


preds = pred_gbt.select('label','probability').rdd.map(lambda row: (float(row['probability'][1]), float(row['label'])))
points = CurveMetrics(preds).get_curve('roc')


# In[67]:


plt.figure()
x_val = [x[0] for x in points]
y_val = [x[1] for x in points]
plt.title('Gradient Boosting Classifier - ROC Curve')
plt.plot(x_val, y_val)


# In[ ]:





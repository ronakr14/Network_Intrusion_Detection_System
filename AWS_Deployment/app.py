import numpy as np
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('Random_Forest_Classifier_FS_DTC.pkl', 'rb'))

@app.route('/') #index or landing page of website
def home():
    return render_template('index.html')
# 127.0.0.1:8080/predict
@app.route('/predict',methods=['POST']) #post method is used to send parameters in http request
def predict():
    '''
    For rendering results on HTML GUI'''
    features = [int(x) for x in request.form.values()]
    features=np.array(features)
    features=features.reshape(1,-1)
    scaler=StandardScaler()
    features=scaler.fit_transform(features)
    features=np.insert(features,9,1)
    features=np.insert(features,12,1)
    features=np.insert(features,14,1)
    features=np.insert(features,16,1)
    features=np.insert(features,17,1)
    features=np.insert(features,19,1)
    features=np.insert(features,20,1)
    features=np.insert(features,21,1)
    features=np.insert(features,22,1)
    features=np.insert(features,23,1)
    features=np.insert(features,25,1)
    final_features=features.reshape(1,-1)
    print(final_features)
    print("Features are ", final_features, " No of features =", len(final_features))
    prediction = model.predict(final_features)
    if prediction==1:
        output="Malign"
    else:
        output="Benign"
    # output =10
    return render_template('index.html', 
    	Classification_text='User/Intruder is {}'
        .format(output))


if __name__ == "__main__":
    app.run(host="0.0.0.0",port=5000)
    #app.run(host="127.0.0.1",port=8080)
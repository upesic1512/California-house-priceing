import json
import pickle

from flask import Flask,request,app,jsonify,url_for,render_template
import matplotlib
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd


app = Flask(__name__)
regmodel=pickle.load(open('regmodel.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods = ['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    sc_Y = StandardScaler()
    Y = sc_Y.fit_transform(np.array(list(data.values())).reshape(1.-1))
    new_data = sc_Y.transform(Y)
    output = regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict', methods = ['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    sc_X = StandardScaler()
    X= sc_X.fit_transform(np.array(data).reshape(1,-1))
    final_input = sc_X.transform(X)
    print(final_input)
    output = regmodel.predict(final_input)[0]
    return render_template("home.html", prediction_text = "The house price p prediction is {}".format(output))



if __name__ == '__main__':
    app.run(debug = True)

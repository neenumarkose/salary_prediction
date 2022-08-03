# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 16:54:30 2022

@author: noopa
"""


import numpy as np
import pickle
import pandas as pd
from flask import Flask, request
from flask import Flask, request, jsonify, render_template

app=Flask(__name__)
pickle_in = open("model.pkl","rb")
classifier=pickle.load(pickle_in)

@app.route('/')
def home():
    return render_template('index.html')



@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()]
    final_features = [np.array(int_features, dtype=float)]
    prediction = classifier.predict(final_features)
    output = round(prediction[0], 0)
    
    return render_template('index.html', prediction_text='The salary predicted is {}'.format(output))
    
    


if __name__=='__main__':
    app.run()
from flask import Flask,request,jsonify,render_template
from flask_cors import cross_origin
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__, template_folder="templates")
model = pickle.load(open("./models/SVC_Classifier.pkl", "rb"))
@app.route("/",methods=['GET'])
@cross_origin()
def home():
    return render_template("index.html")

@app.route("/predict",methods=['GET', 'POST'])
@cross_origin()
def predict():
    if request.method == "POST":
        # Pregnancies
        variance=float(request.form['variance'])
        # Glucose
        skewness = float(request.form['skewness'])
        # bloodpresure
        curtosis= float(request.form['curtosis'])
        # SkinThickness
        entropy = float(request.form['entropy'])
        
        input_lst = [variance , skewness , curtosis , entropy ]
        pred = model.predict([input_lst])
        output = pred
        print(output)
        if output == 0:
            return render_template('index.html',prediction_texts="The aunthentication is not safe.!!!. ")
        else:
            return render_template('index.html',prediction_text="The authentication is safe...")
    return render_template("index.html")

if __name__=='__main__':
    app.run(debug=True)

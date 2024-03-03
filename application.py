from flask import Flask,render_template,request,redirect
#from flask_cors import CORS,cross_origin   if you unable to deploy in herooku use this 'CORS'
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
model =  pickle.load(open("LinearRegressionModel.pkl", 'rb'))
car= pd.read_csv('Cleaned Car.csv')
@app.route('/')
def index():
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    year = sorted(car['year'].unique(), reverse=True)
    fuel_type = sorted(car['fuel_type'].unique())
    companies.insert(0,"Select Company")
    return render_template('index.html', companies = companies, car_models= car_models, years=year, fuel_types=fuel_type)

@app.route('/predict', methods= ['POST'])
def predict():
    company = request.form.get('company')
    car_models = request.form.get('car_models')
    year = int(request.form.get('year'))
    fuel_types= request.form.get('fuel_type')
    kms_driven = int(request.form.get('kilo_driven'))
    print(company,car_models, year,fuel_types,kms_driven )

    prediction = model.predict(pd.DataFrame([[car_models, company, year, kms_driven, fuel_types]],  columns=['name', 'company', 'year', 'kms_driven', 'fuel_type']))


    return str(np.round(prediction[0],2))

if __name__=="__main__":
    app.run(debug=True)
    #app.run(host='0.0.0.0')

from flask import Flask, render_template, request
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from model import predic_xgb
from pymode import predic_rf
import warnings
app = Flask(__name__,template_folder='templates',static_folder="static")

import pickle
model = pickle.load(open('stacked_model1.pkl', 'rb'))

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/form1")
def form():
    return render_template('form.html')

@app.route("/form")
def form_rf():
    return render_template('new.html')

#@app.route("/form2")
#def form_st():
    # return render_template('stpred.html')

@app.route("/DataAnalysis")
def analysis():
    return render_template('churnprofile01.html')

@app.route("/Analysis")
def dataanalysis():
    return render_template('sample.html')

@app.route("/About Us")
def about():
    return render_template('about.html')

@app.route('/pred', methods=['POST'])
def pred_rf():
    return predic_rf()

@app.route('/predict', methods=['POST'])
def predict():
    return predic_xgb()

@app.route('/predictions', methods=['GET','POST'])
def predictions():
    if request.method == "POST":
     gender = request.form['Gender']
     senior_citizen = request.form['Senior_Citizen']
     tenure_months = request.form['Tenure_Months']
     phone_service = request.form['Phone_Service']
     internet_service = request.form['Internet_Service']
     streaming_tv = request.form['Streaming_TV']
     streaming_movies = request.form['Streaming_Movies']
     contract = request.form['Contract']
     payment_method = request.form['Payment_Method']
     monthly_charges = request.form['Monthly_Charges']
     total_charges = request.form['Total_Charges']

     new_data = pd.DataFrame({'Gender': gender, 
                          'Senior_Citizen': senior_citizen, 
                          'Tenure_Months': tenure_months,
                          'Phone_Service': phone_service,
                          'Internet_Service': internet_service,
                          'Streaming TV': streaming_tv,
                          'Streaming Movies': streaming_movies,
                          'Contract': contract,
                          'Payment Method': payment_method,
                          'Monthly Charges': monthly_charges,
                          'Total_Charges': total_charges}, index=[0])

     new_data['Gender'] = new_data['Gender'].map({'Male': 0, 'Female': 1})
     new_data['Senior_Citizen'] = new_data['Senior_Citizen'].map({'Yes': 1, 'No': 0})
     new_data['Phone_Service'] = new_data['Phone_Service'].map({'Yes': 1, 'No': 0})
     new_data['Internet_Service'] = new_data['Internet_Service'].map({'No': 2, 'DSL': 0, 'Fiber optic': 1})
     new_data['Contract'] = new_data['Contract'].map({'Month-to-month': 0, 'One year': 2, 'Two year': 1})
     new_data['Payment Method'] = new_data['Payment Method'].map({'Mailed check': 0, 'Credit card (automatic)': 3, 'Bank transfer (automatic)': 2, 'Electronic check': 1})
     new_data['Streaming TV'] = new_data['Streaming TV'].map({'Yes': 1, 'No': 0})
     new_data['Streaming Movies'] = new_data['Streaming Movies'].map({'Yes': 1, 'No': 0})

     # Get the list of feature names used in the StackingClassifier
     cols = []
     for estimator in model.estimators_:
            if hasattr(estimator, "get_feature_names"):
                cols.extend(estimator.get_feature_names())
            else:
                cols.extend(estimator._get_param_names())
     cols = list(set(cols))
     # Assuming `X` is your feature matrix with column names
     telecom=pd.read_csv("training_data.csv")
     telecom.drop("City", axis=1, inplace=True)

     X = telecom.drop('Churn', axis=1)
     for clf in model.estimators_:
       try:
          clf.feature_names_in_ = X.columns
       except AttributeError:
          pass

     new_prediction = model.predict(new_data)
     prob = model.predict_proba(new_data)[0][1]
     message= 100-float(round(prob * 100, 2))
     return render_template("stpred.html",message=message)
    else:
     return render_template("stpred.html")


if __name__ == '__main__':
 app.run(debug=True)

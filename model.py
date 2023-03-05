import pandas as pd
import numpy as np
import warnings
import seaborn as sns
from flask import Flask, request, render_template
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score


def predic_xgb():
    telecom=pd.read_csv("training_data.csv")
    telecom.head()
    # telecom.info()
    telecom.isnull().sum()

    telecom['Total_Charges'] = pd.to_numeric(telecom['Total_Charges'],errors='coerce')
    nan_cols = [i for i in telecom.columns if telecom[i].isnull().any()]
    nan_cols
    telecom.info()
    telecom.shape
    duplicateRows = telecom[telecom.duplicated()]
    duplicateRows
    def cat_unique_col_values(df):
        for column in df:
            if df[column].dtypes=='object':
                print(f'{column}: {df[column].unique()}')
    telecom.replace("No internet service","No",inplace=True)
    cat_unique_col_values(telecom)

    telecom.shape
    telecom.drop("City", axis=1, inplace=True)
    # telecom.tail(10)

    for col in telecom.columns:
        print(f"{col}: {telecom[col].nunique()} unique values")

    for col in telecom.columns:
        if telecom[col].dtype == 'object':
            telecom[col], _ = pd.factorize(telecom[col])


    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import LabelEncoder
    from sklearn.linear_model import LogisticRegression

    for col in telecom.columns:
        print(f"{col}: {telecom[col].nunique()} unique values")

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix

    telecom.dropna(inplace=True)

    # Split data into training and testing sets
    X = telecom.drop('Churn', axis=1)
    # print(X)
    y = telecom['Churn']
    print(y)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
    # from sklearn.ensemble import RandomForestClassifier

    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    #from sklearn.preprocessing import StandardScaler
    #sc = StandardScaler()
    #X_train = sc.fit_transform(x_train)
    #X_test = sc.transform(x_test)

    # Build the XGBoost model
    model = xgb.XGBClassifier(n_jobs=-1, random_state=123)
    model.fit(x_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(x_test)

    # Evaluate the model performance
    from sklearn.metrics import accuracy_score, confusion_matrix

    print("Accuracy:", accuracy_score(y_test, y_pred))
    confusion = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", confusion)
    from sklearn.model_selection import RandomizedSearchCV
    # XGBoost clasifier
    from xgboost import XGBClassifier
    xg = XGBClassifier(objective='binary:logistic')
    param_grid={
    'learning_rate':[1,0.5,0.1,0.01],
    'max_depth':[3,5,10,20],
    'n_estimators':[10,50,100,200]
    }
    grid = RandomizedSearchCV(XGBClassifier(objective='binary:logistic'), param_grid, verbose=3)
    grid.fit(x_train, y_train)
    grid.best_params_
    xg = XGBClassifier(n_estimators= 100, max_depth= 5, learning_rate= 0.1)
    xg.fit(x_train, y_train)
    # checking accuracy of test dataset
    print("testing accuracy is : ", xg.score(x_test, y_test)*100)


    # Get input data from user
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

    # Create a DataFrame from input data
    new_data = pd.DataFrame({'Gender': gender, 
                          'Senior_Citizen': senior_citizen, 
                          'Tenure_Months': float(tenure_months),
                          'Phone_Service': phone_service,
                          'Internet_Service': internet_service,
                          'Streaming TV': streaming_tv,
                          'Streaming Movies': streaming_movies,
                          'Contract': contract,
                          'Payment Method': payment_method,
                          'Monthly Charges': float(monthly_charges),
                          'Total_Charges': float(total_charges)}, index=[0])

    # Map categorical columns
    new_data['Gender'] = new_data['Gender'].map({'Male': 0, 'Female': 1})
    new_data['Senior_Citizen'] = new_data['Senior_Citizen'].map({'Yes': 1, 'No': 0})
    new_data['Phone_Service'] = new_data['Phone_Service'].map({'Yes': 1, 'No': 0})
    new_data['Internet_Service'] = new_data['Internet_Service'].map({'No': 2, 'DSL': 0, 'Fiber optic': 1})
    new_data['Contract'] = new_data['Contract'].map({'Month-to-month': 0, 'One year': 2, 'Two year': 1})
    new_data['Payment Method'] = new_data['Payment Method'].map({'Mailed check': 0, 'Credit card (automatic)': 3, 'Bank transfer (automatic)': 2, 'Electronic check': 1})
    new_data['Streaming TV'] = new_data['Streaming TV'].map({'Yes': 1, 'No': 0})
    new_data['Streaming Movies'] = new_data['Streaming Movies'].map({'Yes': 1, 'No': 0})



    # Predict target variable
    new_prediction = model.predict(new_data)
    prob = model.predict_proba(new_data)[0][1]
    message= 100-float(round(prob * 100, 2))
    # if new_prediction[0]==0:
    #     message='He will leave.'
    # else:
    #     message='He will stay.'
    # Return prediction as a string
    return render_template('form.html', message=message)

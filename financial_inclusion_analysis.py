# Installing necessary packages
# Note: This is a comment for reference; run these in your terminal or environment
# pip install pandas numpy ydata-profiling scikit-learn streamlit

# Importing required libraries
import pandas as pd
import numpy as np
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import streamlit as st

# Loading the dataset
df = pd.read_csv('Financial_inclusion_dataset.csv')

# Displaying general information about the dataset
print("Dataset Info:")
print(df.info())
print("\nFirst 5 rows of the dataset:")
print(df.head())
print("\nSummary statistics:")
print(df.describe(include='all'))

# Creating a pandas profiling report
profile = ProfileReport(df, title="Financial Inclusion Profiling Report", explorative=True)
profile.to_file("financial_inclusion_report.html")

# Handling missing and corrupted values
print("\nMissing values:")
print(df.isnull().sum())
# No missing values based on provided data, but adding check for completeness
df = df.dropna()  # Drop rows with missing values if any

# Checking and removing duplicates
print("\nNumber of duplicate rows:", df.duplicated().sum())
df = df.drop_duplicates()

# Handling outliers for numerical columns (household_size, age_of_respondent)
# Using IQR method to detect and cap outliers
def cap_outliers(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return series.clip(lower=lower_bound, upper=upper_bound)

df['household_size'] = cap_outliers(df['household_size'])
df['age_of_respondent'] = cap_outliers(df['age_of_respondent'])

# Encoding categorical features
le = LabelEncoder()
categorical_columns = ['country', 'bank_account', 'location_type', 'cellphone_access', 
                      'gender_of_respondent', 'relationship_with_head', 'marital_status', 
                      'education_level', 'job_type']

# Creating a dictionary to store label encoders for Streamlit app
label_encoders = {}
for col in categorical_columns:
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Preparing features and target
X = df.drop(['bank_account', 'uniqueid', 'year'], axis=1)  # Dropping uniqueid and year as they are not predictive
y = df['bank_account']

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training a RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluating the model
y_pred = model.predict(X_test)
print("\nModel Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Saving the model for Streamlit app
import joblib
joblib.dump(model, 'financial_inclusion_model.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')

# Streamlit application
st.title("Financial Inclusion Prediction App")

# Creating input fields for features
country = st.selectbox("Country", label_encoders['country'].classes_)
location_type = st.selectbox("Location Type", label_encoders['location_type'].classes_)
cellphone_access = st.selectbox("Cellphone Access", label_encoders['cellphone_access'].classes_)
household_size = st.number_input("Household Size", min_value=1, max_value=20, value=3)
age_of_respondent = st.number_input("Age of Respondent", min_value=16, max_value=100, value=30)
gender_of_respondent = st.selectbox("Gender", label_encoders['gender_of_respondent'].classes_)
relationship_with_head = st.selectbox("Relationship with Head", label_encoders['relationship_with_head'].classes_)
marital_status = st.selectbox("Marital Status", label_encoders['marital_status'].classes_)
education_level = st.selectbox("Education Level", label_encoders['education_level'].classes_)
job_type = st.selectbox("Job Type", label_encoders['job_type'].classes_)

# Creating a button for prediction
if st.button("Predict"):
    # Loading the model and encoders
    model = joblib.load('financial_inclusion_model.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    
    # Preparing input data
    input_data = {
        'country': label_encoders['country'].transform([country])[0],
        'location_type': label_encoders['location_type'].transform([location_type])[0],
        'cellphone_access': label_encoders['cellphone_access'].transform([cellphone_access])[0],
        'household_size': household_size,
        'age_of_respondent': age_of_respondent,
        'gender_of_respondent': label_encoders['gender_of_respondent'].transform([gender_of_respondent])[0],
        'relationship_with_head': label_encoders['relationship_with_head'].transform([relationship_with_head])[0],
        'marital_status': label_encoders['marital_status'].transform([marital_status])[0],
        'education_level': label_encoders['education_level'].transform([education_level])[0],
        'job_type': label_encoders['job_type'].transform([job_type])[0]
    }
    
    # Creating DataFrame for prediction
    input_df = pd.DataFrame([input_data])
    
    # Making prediction
    prediction = model.predict(input_df)[0]
    prediction_label = label_encoders['bank_account'].inverse_transform([prediction])[0]
    
    st.write(f"Prediction: The individual is {'likely' if prediction_label == 'Yes' else 'unlikely'} to have a bank account.")
```python
import streamlit as st
import pandas as pd
import joblib

# Load the model and label encoders
try:
    pickle_in = open('financial_inclusion_xgboost_model.pkl', 'rb')
    model = joblib.load(pickle_in)
    pickle_in = open('label_encoders.pkl', 'rb')
    label_encoders = joblib.load(pickle_in)
    model = joblib.load('financial_inclusion_xgboost_model.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
except FileNotFoundError:
    st.error("Model or label encoders file not found. Please ensure 'financial_inclusion_xgboost_model.pkl' and 'label_encoders.pkl' are in the directory.")
    st.stop()

# Streamlit application
st.title("Financial Inclusion Prediction")
st.write("Predict whether an individual is likely to have a bank account based on their details.")

# Input fields
country = st.selectbox("Country", label_encoders['country'].classes_)
location_type = st.selectbox("Location Type", label_encoders['location_type'].classes_)
cellphone_access = st.selectbox("Cellphone Access", label_encoders['cellphone_access'].classes_)
household_size = st.number_input("Household Size", min_value=1, max_value=20, value=3)
age_of_respondent = st.number_input("Age", min_value=16, max_value=100, value=30)
gender = st.selectbox("Gender", label_encoders['gender_of_respondent'].classes_)
relationship = st.selectbox("Relationship with Head", label_encoders['relationship_with_head'].classes_)
marital_status = st.selectbox("Marital Status", label_encoders['marital_status'].classes_)
education = st.selectbox("Education Level", label_encoders['education_level'].classes_)
job_type = st.selectbox("Job Type", label_encoders['job_type'].classes_)

# Prediction button
if st.button("Predict"):
    # Prepare input data
    input_data = {
        'country': label_encoders['country'].transform([country])[0],
        'location_type': label_encoders['location_type'].transform([location_type])[0],
        'cellphone_access': label_encoders['cellphone_access'].transform([cellphone_access])[0],
        'household_size': household_size,
        'age_of_respondent': age_of_respondent,
        'gender_of_respondent': label_encoders['gender_of_respondent'].transform([gender])[0],
        'relationship_with_head': label_encoders['relationship_with_head'].transform([relationship])[0],
        'marital_status': label_encoders['marital_status'].transform([marital_status])[0],
        'education_level': label_encoders['education_level'].transform([education])[0],
        'job_type': label_encoders['job_type'].transform([job_type])[0]
    }
    
    # Create DataFrame for prediction
    input_df = pd.DataFrame([input_data])
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    prediction_label = label_encoders['bank_account'].inverse_transform([prediction])[0]
    
    # Display result
    st.success(f"The individual is {'likely' if prediction_label == 'Yes' else 'unlikely'} to have a bank account.")
```

### Required Files for Deployment

In addition to `app.py`, you need:
- `financial_inclusion_xgboost_model.pkl`: The trained XGBoost model.
- `label_encoders.pkl`: The saved label encoders.
- `requirements.txt`: A file listing dependencies.

### Create `requirements.txt`

<xaiArtifact artifact_id="78f70280-100d-4c99-857e-0791a9e70fca" artifact_version_id="80b39c11-4b79-415c-a6d6-f7c64726dc94" title="requirements.txt" contentType="text/plain">
```text
streamlit==1.38.0
pandas==2.2.3
numpy==2.1.1
scikit-learn==1.5.2
xgboost==2.1.1
joblib==1.4.2
```

### Steps to Generate Model Files

If you haven’t generated `financial_inclusion_xgboost_model.pkl` and `label_encoders.pkl`, use the following script (`train_model.py`) to create them. Run this locally with `Financial_inclusion_dataset.csv` in your directory.

<xaiArtifact artifact_id="f1832aee-6d33-4c4b-8f97-e4fb784a6f37" artifact_version_id="4d40745b-0d64-4638-8548-9a4877cfc264" title="train_model.py" contentType="text/python">
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import joblib

# Load dataset
df = pd.read_csv('Financial_inclusion_dataset.csv')

# Preprocess
df = df.dropna().drop_duplicates()

# Cap outliers
def cap_outliers(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    return series.clip(lower=Q1 - 1.5 * IQR, upper=Q3 + 1.5 * IQR)

df['household_size'] = cap_outliers(df['household_size'])
df['age_of_respondent'] = cap_outliers(df['age_of_respondent'])

# Encode categorical features
categorical_columns = ['country', 'bank_account', 'location_type', 'cellphone_access', 
                      'gender_of_respondent', 'relationship_with_head', 'marital_status', 
                      'education_level', 'job_type']
label_encoders = {}
le = LabelEncoder()
for col in categorical_columns:
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Prepare features and target
X = df.drop(['bank_account', 'uniqueid', 'year'], axis=1)
y = df['bank_account']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Save model and encoders
joblib.dump(model, 'financial_inclusion_xgboost_model.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
print("Model and encoders saved.")
```

Run it:
```bash
python train_model.py
```

### Deployment Steps

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Test Locally**:
   Ensure `app.py`, `financial_inclusion_xgboost_model.pkl`, and `label_encoders.pkl` are in your directory, then run:
   ```bash
   streamlit run app.py
   ```
   Open `http://localhost:8501` to test the app.

3. **Create GitHub Repository**:
   - Create a new GitHub repository (e.g., `financial-inclusion-app`).
   - Add `app.py`, `financial_inclusion_xgboost_model.pkl`, `label_encoders.pkl`, and `requirements.txt`.
   - Push to GitHub:
     ```bash
     git init
     git add app.py financial_inclusion_xgboost_model.pkl label_encoders.pkl requirements.txt
     git commit -m "Streamlit app for financial inclusion"
     git remote add origin <your-repo-url>
     git push -u origin main
     ```

4. **Deploy on Streamlit Community Cloud**:
   - Go to [Streamlit Community Cloud](https://streamlit.io/cloud).
   - Click "New app", select your repository, branch (e.g., `main`), and set `app.py` as the main script.
   - Deploy. You’ll get a URL (e.g., `https://your-app.streamlit.app`).

### Notes
- **File Size**: Ensure `financial_inclusion_xgboost_model.pkl` is <100MB for GitHub.
- **Errors**: Check Streamlit Cloud logs if deployment fails (e.g., missing files).
- **Dataset**: `Financial_inclusion_dataset.csv` is not needed for deployment.

This `app.py` provides a simple, deployable Streamlit app for predicting financial inclusion. Let me know if you need help with deployment or tweaks!

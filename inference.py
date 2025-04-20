import pickle
import numpy as np
import pandas as pd

# Load model
with open('UTS_Model_Deployment/model_xgb.pkl', 'rb') as f:
    model = pickle.load(f)

# Load scaler
with open('UTS_Model_Deployment/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Input Asli yang belum di preprocess
raw_input = {
    'person_gender': 'male',
    'person_education': 'Master',
    'person_income': 46467.0,
    'person_emp_exp': 5,
    'person_home_ownership': 'RENT',
    'loan_amnt': 12000.0,
    'loan_intent': 'PERSONAL',
    'loan_int_rate': 13.23,
    'loan_percent_income': 0.26,
    'cb_person_cred_hist_length': 3.0,
    'credit_score': 659,
    'previous_loan_defaults_on_file': 'No'
}

# maps untuk encode categorical variable
gender_map = {'female': 0, 'male': 1}
education_map = {'High School': 0, 'Associate': 1, 'Bachelor': 2, 'Master': 3, 'Doctorate': 4}
home_ownership_map = {'RENT': 0, 'OWN': 1, 'MORTGAGE': 2, 'OTHER': 3}
loan_intent_map = {'PERSONAL': 0, 'EDUCATION': 1, 'MEDICAL': 2, 'VENTURE': 3, 'HOMEIMPROVEMENT': 4, 'DEBTCONSOLIDATION': 5}
default_map = {'No': 0, 'Yes': 1}

# encode variable categorical agar bisa di predict oleh model
encoder = {
    'person_gender': gender_map[raw_input['person_gender']],
    'person_education': education_map[raw_input['person_education']],
    'person_home_ownership': home_ownership_map[raw_input['person_home_ownership']],
    'loan_intent': loan_intent_map[raw_input['loan_intent']],
    'previous_loan_defaults_on_file': default_map[raw_input['previous_loan_defaults_on_file']]
}

# re define column apa saja yang mau di scale
numerical_cols = ['person_income', 'person_emp_exp', 'loan_amnt', 'loan_int_rate',
                'loan_percent_income', 'cb_person_cred_hist_length', 'credit_score']

numerical_values = pd.DataFrame([{
    col: raw_input[col] for col in numerical_cols
}])

# Scale numerical numerical valuesnya so nanti sesuai dengan yg di train
numerical_scaled = scaler.transform(numerical_values)

# Gabung encoded + scaled dgn mempertahankan urutan biar sesuai dengan model yang di train di ipynb
final_input = np.array([
    encoder['person_gender'],
    encoder['person_education'],
    numerical_scaled[0][0], # income
    numerical_scaled[0][1], # emp_exp
    encoder['person_home_ownership'],
    numerical_scaled[0][2], # loan_amnt
    encoder['loan_intent'],
    *numerical_scaled[0][3:], # ini sisanya, pake * biar ga usah di tulis satu-satu dan urut jg
    encoder['previous_loan_defaults_on_file']
], dtype=float).reshape(1, -1)

# predictt
prediction = model.predict(final_input)
print("Prediction:", prediction)

import streamlit as st
import pickle
import pandas as pd
import os

st.set_page_config(page_title="Loan Approval Predictor", page_icon="💸", layout="centered") # ini lucu juga pake icon

# Ini buat ngambil file pickle dan performance dengan cache
@st.cache_resource
def load_pickle(filename):
    try:
        with open(os.path.join(BASE_DIR, filename), 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error(f"Error: {filename} not found in {BASE_DIR}")
        raise
    except Exception as e:
        st.error(f"Error loading {filename}: {str(e)}")
        raise

# directory base karena filenya ada di dalam folder yang sama
# TODO: ini juga bisa diubah sesuai kebutuhan, misal mau di folder lain
BASE_DIR = os.path.dirname(__file__)

# Load model yang udah di train, scalernya, sama mapping buat encode input dari user nanti
model = load_pickle('model_xgb.pkl') # bismilah model 93% accuracy
scaler = load_pickle('scaler.pkl')
mappings = load_pickle('mappings.pkl')

# Mappings for categorical variables
gender_map = mappings.get('gender_map', {})  # Fallback if missing
education_map = mappings.get('education_map', {})
home_ownership_map = mappings.get('home_ownership_map', {})
loan_intent_map = mappings.get('loan_intent_map', {})
default_map = mappings.get('default_map', {})

# Redeclare numerical valius buat scaling lagi nanti
numerical_cols = [
    'person_income', 'person_emp_exp', 'loan_amnt', 'loan_int_rate',
    'loan_percent_income', 'cb_person_cred_hist_length', 'credit_score'
]

#! Urutan dari kolom pada streamlitnya kaya gimana
expected_columns = [
    'person_gender', 'person_education', 'person_income', 'person_emp_exp',
    'person_home_ownership', 'loan_amnt', 'loan_intent', 'loan_int_rate',
    'loan_percent_income', 'cb_person_cred_hist_length', 'credit_score',
    'previous_loan_defaults_on_file'
]

# The header for the application of streamlit
st.title("Loan Approval Predictor 💸")
st.write("Enter the details below to predict loan approval. Use realistic values for accurate results.")
st.write("With this application, you can predict if your load application will be approved or not and save your time!")

# Input fields
# values : ini untuk initial value di streamlitnya, 
# min_value : ini untuk minimum value yang bisa dimasukin kaya misalnya 0 untuk beberapa case
# max_value : ini untuk maximum value yang bisa dimasukin kaya credit score itu gabisa diatas 850
gender = st.selectbox("Gender", list(gender_map.keys()))
education = st.selectbox("Last Education", list(education_map.keys()), index = 3) # jadiin master as default
income = st.number_input("Annual Income (In USD)", min_value=0, value=50000)
emp_exp = st.number_input("Employment Experience (In Years)", min_value=0, value=5)
home_ownership = st.selectbox("Home Ownership", list(home_ownership_map.keys()))
loan_amount = st.number_input("Loan Amount (In USD)", min_value=0,max_value=200000,value=12000) #kasih warning kalau diatas 200 ribu
loan_intent = st.selectbox("Intention of Loan", list(loan_intent_map.keys()))
interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, value=13.23, format="%.2f")
credit_hist_length = st.number_input("Credit History Length (In Years)", min_value=0, value=3)
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=659, step=1)
prev_defaults = st.selectbox("Previous Loan Default", list(default_map.keys()))

# Ini biar nanti user ga bingung sama inputnya, agaknya ribet juga kalau harus itung percentage sendiri hehe
loan_percent_income = loan_amount / income if income > 0 else 0.0

# Prediction logic
if st.button("Predict Loan Approval"):
    # Validate inputs dan error handling
    # Ini buat ngecek apakah inputan user valid atau engga, misalnya income ga boleh 0, loan amount ga boleh 0, dll
    if income <= 0 or loan_amount <= 0:
        st.error("Please enter valid values for Income and Loan Amount (greater than 0).")
    elif emp_exp < 0 or credit_hist_length < 0:
        st.error("Employment Experience and Credit History Length cannot be negative.")
    elif credit_score < 300 or credit_score > 850:
        st.error("Credit Score must be between 300 and 850.")
    elif loan_percent_income > 1.0:
        st.error("Loan Amount cannot exceed 100% of your Annual Income.")
    else:
        with st.spinner("Predicting..."):
            # Input dataframe
            input_data = pd.DataFrame([{
                'person_gender': gender_map[gender],
                'person_education': education_map[education],
                'person_income': income,
                'person_emp_exp': emp_exp,
                'person_home_ownership': home_ownership_map[home_ownership],
                'loan_amnt': loan_amount,
                'loan_intent': loan_intent_map[loan_intent],
                'loan_int_rate': interest_rate,
                'loan_percent_income': loan_percent_income,
                'cb_person_cred_hist_length': credit_hist_length,
                'credit_score': credit_score,
                'previous_loan_defaults_on_file': default_map[prev_defaults]
            }])

            # Ensure correct feature order
            try:
                input_data = input_data[expected_columns]
            except KeyError as e:
                st.error(f"Feature mismatch: {str(e)}. Please check model compatibility.")
                st.stop()

            # Scaler biar sesuai sama data yang dimasukin ke model xgb
            try:
                input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])
            except Exception as e:
                st.error(f"Scaling failed: {str(e)}. Please check scaler compatibility.")
                st.stop()

            # Prediction inference internally di streamlit
            try:
                prediction = model.predict(input_data)[0]
                proba = model.predict_proba(input_data)[0][1]
                if prediction == 1:
                    st.success(f"🎉 Congratulations! Your loan is likely to be approved. \n \n (Probability of Approval: {proba*100:.2f}%)")
                    st.balloons()
                else:
                    st.error(f"❌ Unfortunately, your loan application is likely to be rejected. \n \n (Probability of Approval: {proba*100:.2f}%)")

            # for error handling, misalnya kalo modelnya ga sesuai sama inputan gt gt
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}. Please check model compatibility.")


st.markdown("---")
st.subheader("🔍 Test Cases")

st.write("Click the buttons below to see how the predictions on different test cases, as your reference")
col1, col2 = st.columns(2)

with col1:
    if st.button("Test Case 1: Likely Approved"):
        test_data = pd.DataFrame([{
            'person_gender': gender_map.get('male'),
            'person_education': education_map.get('Associate'),
            'person_income': 83873,
            'person_emp_exp': 6,
            'person_home_ownership': home_ownership_map.get('RENT'),
            'loan_amnt': 15000,
            'loan_intent': loan_intent_map.get('HOMEIMPROVEMENT'),
            'loan_int_rate': 16.7,
            'loan_percent_income': 15000 / 83873,
            'cb_person_cred_hist_length': 3,
            'credit_score': 627,
            'previous_loan_defaults_on_file': default_map.get('No')
        }])

        # ini dibuat agar test casenya dilihat bisa lewat string ga lewat angka hasil mapping encoding aj, isinnya sama persis
        inputforjson = { 
            'person_gender': 'male',
            'person_education': 'Associate',
            'person_income': 83873,
            'person_emp_exp': 6,
            'person_home_ownership': 'RENT',
            'loan_amnt': 15000,
            'loan_intent': 'HOMEIMPROVEMENT',
            'loan_int_rate': 16.7,
            'loan_percent_income': 15000 / 83873,
            'cb_person_cred_hist_length': 3,
            'credit_score': 627,
            'previous_loan_defaults_on_file': 'No'
        }

        st.write("You can see the test case input belom as reference")
        # st.json(st.json(test_data.to_dict(orient="records")[0])) 
        st.json(inputforjson)
        try:
            test_data[numerical_cols] = scaler.transform(test_data[numerical_cols])
            prediction = model.predict(test_data)[0]
            proba = model.predict_proba(test_data)[0][1]

            st.info("**Test Case 1 Result**")
            if prediction == 1:
                st.success(f"✅ Loan Approved! (Probability: {proba*100:.2f}%)")
            else:
                st.error(f"❌ Loan Rejected! (Probability: {proba*100:.2f}%)")
        except Exception as e:
            st.error(f"Error in Test Case 1: {str(e)}")

with col2:
    if st.button("Test Case 2: Likely Rejected"):
        test_data = pd.DataFrame([{
            'person_gender': gender_map.get('Female', 1),
            'person_education': education_map.get('High School'),
            'person_income': 18000,
            'person_emp_exp': 1,
            'person_home_ownership': home_ownership_map.get('RENT'),
            'loan_amnt': 16000,
            'loan_intent': loan_intent_map.get('PERSONAL'),
            'loan_int_rate': 22.0,
            'loan_percent_income': 16000 / 18000,
            'cb_person_cred_hist_length': 1,
            'credit_score': 510,
            'previous_loan_defaults_on_file': default_map.get('Yes')
        }])

        inputforjson = {
            'person_gender': gender_map.get('Female', 1),
            'person_education': 'High School',
            'person_income': 18000,
            'person_emp_exp': 1,
            'person_home_ownership':'RENT',
            'loan_amnt': 16000,
            'loan_intent': 'PERSONAL',
            'loan_int_rate': 22.0,
            'loan_percent_income': 16000 / 18000,
            'cb_person_cred_hist_length': 1,
            'credit_score': 510,
            'previous_loan_defaults_on_file': 'Yes'
        }

        st.write("You can see the test case input belom as reference")
        st.json(inputforjson)
        try:
            test_data[numerical_cols] = scaler.transform(test_data[numerical_cols])
            prediction = model.predict(test_data)[0]
            proba = model.predict_proba(test_data)[0][1]

            st.info("**Test Case 2 Result**")
            if prediction == 1:
                st.success(f"✅ Loan Approved! (Probability: {proba*100:.2f}%)")
            else:
                st.error(f"❌ Loan Rejected! (Probability: {proba*100:.2f}%)")
        except Exception as e:
            st.error(f"Error in Test Case 2: {str(e)}")
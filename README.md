
![Python](https://img.shields.io/badge/Python-3.10-blue)
![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-ff4b4b)

You can visit deployed streamlit on streamlit cloud here!
https://eu5ajqmfmwrwdcjcsmvln3.streamlit.app/

# 🏦 Loan Approval Prediction System
This is a project created for the Mid Term that aims to build a system for predicting loan approval using Machine Learning.
The system is designed to assist banks in filtering out loan applicants who are likely to be rejected and help individuals get a rough idea of their loan approval chances — without having to go to the bank. 🎯

## 📊 Dataset Description
The dataset contains profiles of loan applicants along with whether their loan was approved or rejected.
Key features include:

| Feature                          | Description                                                |
| -------------------------------- | ---------------------------------------------------------- |
| `person_age`                     | Age of the applicant ➡️ *Dropped due to multicollinearity* |
| `person_gender`                  | Gender                                                     |
| `person_education`               | Highest education level                                    |
| `person_income`                  | Annual income                                              |
| `person_emp_exp`                 | Years of employment experience                             |
| `person_home_ownership`          | Home ownership status                                      |
| `loan_amnt`                      | Requested loan amount                                      |
| `loan_intent`                    | Purpose of the loan                                        |
| `loan_int_rate`                  | Interest rate                                              |
| `loan_percent_income`            | Loan as a percentage of income                             |
| `cb_person_cred_hist_length`     | Length of credit history (years)                           |
| `credit_score`                   | Credit score                                               |
| `previous_loan_defaults_on_file` | Indicator for past defaults                                |
| `loan_status` (**target**)       | Loan approval status → `1`: Approved ✅, `0`: Rejected ❌    |


## 🛠️ System Workflow
1. Preprocessing
2. Model Training
3. Model Saving
4. Deployment using streamlit that includes 2 test cases, approved and rejected

## ✨ Features
- Predicts loan approval using applicant data
- Built with XGBoost for high accuracy
- User-friendly interface with Streamlit
- Includes sample test cases for clarity

## 🚀 How to Run the App
Run this command in the terminal of the environment
1. Install Dependencies 
<pre lang="bash"> pip install -r requirements.txt </pre>

2. Run Streamlit
<pre lang="bash"> streamlit run app.py </pre>

Enjoy predicting your loan chances! 💰📉📈
Feel free to fork, star ⭐, or contribute!

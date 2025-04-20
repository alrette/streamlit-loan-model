import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os

class LoanPrediction:
    def __init__ (self, data_path):
        self.data_path = data_path # datapath ini nantinya akan berupa data loan
        self.scaler = RobustScaler()
        self.model = XGBClassifier(uselabelencoder=False, eval_metric='logloss', random_state=42)
        self.mappings = {} # dictionary untuk mapping categorical variables nantinya
        self.numerical_cols = [] # untuk menyimpan nama numerical columns karena dibutuhkan untuk preprocessing sepeerti scaling nantinya

    def load_data(self):
        self.df = pd.read_csv(self.data_path)
        self.df = self.df.dropna(subset=['person_income']) #drop person income yang null seperti di noteboon ml
        self.df = self.df.drop(columns=['person_age']) #drop person age karena multicollinearity dengan experience
    
    def encode_categorical(self):
        gender_map = {'female': 0, 'male': 1}
        education_map = {'High School': 0, 'Associate': 1, 'Bachelor': 2, 'Master': 3, 'Doctorate': 4}
        home_ownership_map = {'RENT': 0, 'OWN': 1, 'MORTGAGE': 2, 'OTHER': 3}
        loan_intent_map = {'PERSONAL': 0, 'EDUCATION': 1, 'MEDICAL': 2, 'VENTURE': 3, 'HOMEIMPROVEMENT': 4, 'DEBTCONSOLIDATION': 5}
        default_map = {'No': 0, 'Yes': 1}

        self.df['person_gender'] = self.df['person_gender'].map(gender_map)
        self.df['person_education'] = self.df['person_education'].map(education_map)
        self.df['person_home_ownership'] = self.df['person_home_ownership'].map(home_ownership_map)
        self.df['loan_intent'] = self.df['loan_intent'].map(loan_intent_map)
        self.df['previous_loan_defaults_on_file'] = self.df['previous_loan_defaults_on_file'].map(default_map)

        # untuk disimpan didalam pickle nanti untuk dideploy nantinya agar categorical variables tetap bisa diencode sesuai dengan yang sudah ditentukan
        # jadi kita bikin dictonary didalam dict self.mappings
        self.mappings = { 
            'gender_map' : gender_map,
            'education_map' : education_map,
            'home_ownership_map' : home_ownership_map,
            'loan_intent_map' : loan_intent_map,
            'default_map' : default_map
        }

    def split_scale_winsor(self):
        # X dan y gausah dijadiin object karena cuma temporary buat spliiting dan nantinya bakal dimasukin lagi ke self train test
        X = self.df.drop(columns=['loan_status']) 
        y = self.df['loan_status']
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        winsor_limit = {'person_emp_exp':0.05}
        for col, limit in winsor_limit.items():
            self.x_train[col] = winsorize(self.x_train[col], limits=[limit, limit])

        # numerical columns defined here manually karena takut clash dengan encoding with mapping
        self.numerical_cols = ['person_income', 'person_emp_exp', 'loan_amnt', 'loan_int_rate', 
                            'loan_percent_income', 'cb_person_cred_hist_length', 'credit_score']

        self.scaler.fit(self.x_train[self.numerical_cols])
        self.x_train[self.numerical_cols] = self.scaler.transform(self.x_train[self.numerical_cols])
        self.x_test[self.numerical_cols] = self.scaler.transform(self.x_test[self.numerical_cols])

    def train_model(self):
        self.model.fit(self.x_train, self.y_train)

    def evaluate_model(self):
        y_pred = self.model.predict(self.x_test)
        print(accuracy_score(self.y_test, y_pred))
        print(classification_report(self.y_test, y_pred))

    def save_everything(self):
        base_path = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_path, 'model_xgb2.pkl')
        scaler_path = os.path.join(base_path, 'scaler2.pkl')
        mappings_path = os.path.join(base_path, 'mappings2.pkl')
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        with open(mappings_path, 'wb') as f:
            pickle.dump(self.mappings, f)
        print("Done and saved")

# main function untuk jalanin program
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "Dataset_A_loan.csv") # gabungin biar lebih enak
    loan_prediction = LoanPrediction(data_path)
    loan_prediction.load_data()
    loan_prediction.encode_categorical()
    loan_prediction.split_scale_winsor()
    loan_prediction.train_model()
    loan_prediction.evaluate_model()
    loan_prediction.save_everything()
    print("Model, scaler, winsorizer, and mappings saved successfully.")




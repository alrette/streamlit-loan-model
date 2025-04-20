# Loan Approval Prediction System

Project Ujian Tengah Semester ini bertujuan untuk membangun sebuah sistem untuk prediksi persetujuan peminjaman uang melalui implementasi machine learning. Sistem ini dirancang untuk membantu bank menyortir calon peminjam yang memiliki kemungkinan besar ditolak pinjamannnya. Projek ini juga bisa membantu masyarakat untuk mengetahui gambaran kasar persetujuan pinjaman mereka tanpa harus datang langsung ke bank.


# Dataset yang digunakan
Dataset yang digunakan adalah dataset yang berisi mengenai profil calon peminjam dan status peminjaman apakah disetujui atau tidak.
Beberapa fiturnya adalah:

person_age = Usia dari orang tersebut  --> Dibuang dalam preprocessing karena menyebabkan multicollinearity
person_gender = Gender dari orang tersebut
person_education = Tingkat pendidikan tertinggi
person_income = Pendapatan tahunan
person_emp_exp = Tahun pengalaman bekerja
person_home_ownership = Status kepemilikan tempat huni
loan_amnt = Jumlah pinjaman yang diminta
loan_intent = Tujuan dari pinjaman
loan_int_rate = Suku bunga pinjaman
loan_percent_income = Jumlah pinjaman sebagai persentase dari pendapatan tahunan
cb_person_cred_hist_length = Durasi kredit dalam tahun
credit_score = Skor kredit dari orang tersebut
previous_loan_defaults_on_file = Indikator tunggakan pinjaman sebelumnya
loan_status (target variable) = **Persetujuan pinjaman**; 1: diterima dan 0: ditolak

# Proses pembentukan sistem
- Task machine learning yaitu : 
    Preprocessing data (missing value handling, encoding, train test split) 
    Melatih dua macam model machine learning yaitu RandomForest dan XGBoost --> dan ditemukan bahwa XGBoost lebih baik
    Menyimpan model terbaik, scaler, dan encoder dalam file pickle
- Deploying sistem menggunakan platform **streamlit**
- Sistem aplikasi dilengkapi dengan 2 test case yang menunjukkan contoh saat pinjaman diterima dan saat pinjaman ditolak

# Cara untuk menjalankan aplikasi
1. Pastikan jika dependencies sudah benar

Install dependencies:
```bash
pip install -requirements.txt

2. Jalankan dengan call
streamlit run app.py
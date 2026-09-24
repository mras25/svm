# Jalankan dengan: streamlit run app_sederhana.py
# Butuh model.pkl dan scaler.pkl di folder yang sama
import pickle

import pandas as pd
import streamlit as st

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("Deteksi Fraud Kartu Kredit")
file = st.file_uploader("Unggah CSV (kolom V1-V28, Time, Amount)", type="csv")

if file:
    data = pd.read_csv(file)
    x_baru = scaler.transform(data[scaler.feature_names_in_])  # scaling dulu, sama seperti saat training
    data["prediksi"] = model.predict(x_baru)                   # 0 = normal, 1 = fraud
    st.dataframe(data)
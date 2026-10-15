import streamlit as st
import pandas as pd
import joblib
import numpy as np

@st.cache_resource
def load_models():
    model = joblib.load('fraud_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_models()

st.title("💳 Kredi Kartı Dolandırıcılık Tespiti (Fraud Detection)")
st.write("Bu uygulama, yapay sinir ağları ve makine öğrenmesi kullanarak şüpheli kredi kartı işlemlerini anında tespit eder.")

st.subheader("İşlem Simülasyonu ve Detayları:")

# Hızlı test için hazır senaryo butonları
col_btn1, col_btn2 = st.columns(2)
with col_btn1:
    normal_test = st.button("🟢 Normal İşlem Örneği Yükle")
with col_btn2:
    fraud_test = st.button("🔴 Şüpheli (Fraud) İşlem Örneği Yükle")

# Varsayılan değerler
default_amount = 50.0
default_v1, default_v2, default_v4, default_v14, default_v17 = 0.0, 0.0, 0.0, 0.0, 0.0

if normal_test:
    default_amount = 45.0
    default_v1, default_v2, default_v4, default_v14, default_v17 = 0.1, -0.2, 0.3, 0.1, -0.1

if fraud_test:
    default_amount = 25000.0  # Çok yüksek tutar
    default_v1, default_v2, default_v4, default_v14, default_v17 = -25.5, 18.4, 8.2, -15.5, -18.1 # Şüpheli PCA bileşenleri

amount = st.number_input("İşlem Tutarı (Amount)", min_value=0.0, max_value=1000000.0, value=default_amount)

st.write("*(Modelin duyarlı olduğu kritik PCA özellikleri)*")
v1 = st.slider("V1 Bileşeni", -30.0, 30.0, default_v1)
v2 = st.slider("V2 Bileşeni", -30.0, 30.0, default_v2)
v4 = st.slider("V4 Bileşeni", -30.0, 30.0, default_v4)
v14 = st.slider("V14 Bileşeni", -30.0, 30.0, default_v14)
v17 = st.slider("V17 Bileşeni", -30.0, 30.0, default_v17)

if st.button("İşlemi Analiz Et"):
    input_data = np.zeros((1, 29))
    
    # PCA bileşenlerini yerleştiriyoruz
    input_data[0, 0] = v1
    input_data[0, 1] = v2
    input_data[0, 3] = v4   
    input_data[0, 13] = v14 
    input_data[0, 16] = v17 
    
    # Tutarı modelin eğtildiği formata (Scaled Amount) dönüştürüyoruz
    scaled_amount = scaler.transform([[amount]])[0][0]
    input_data[0, -1] = scaled_amount # Ölçeklenmiş tutarı son sütuna ekliyoruz
    
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)
    
    if prediction[0] == 1:
        st.error(f"🚨 DİKKAT! Bu işlem ŞÜPHELİ / DOLANDIRICILIK olabilir! (Risk Oranı: %{probability[0][1]*100:.2f})")
    else:
        st.success(f"✅ Bu işlem GÜVENLİ görünmektedir. (Güven Oranı: %{probability[0][0]*100:.2f})")

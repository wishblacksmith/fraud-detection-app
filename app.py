import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Modelimizi ve scaler'ımızı yüklüyoruz
@st.cache_resource
def load_models():
    model = joblib.load('fraud_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_models()

st.title("💳 Kredi Kartı Dolandırıcılık Tespiti (Fraud Detection)")
st.write("Bu uygulama, yapay sinir ağları ve makine öğrenmesi kullanarak şüpheli kredi kartı işlemlerini anında tespit eder.")

st.subheader("İşlem Detaylarını Girin:")

# Kullanıcıdan temel verileri alalım
amount = st.number_input("İşlem Tutarı (Amount)", min_value=0.0, max_value=100000.0, value=150.0)

# Görsel amaçlı ek kullanıcı girdileri (V1 ve V2 gibi temel PCA bileşenleri)
col1, col2 = st.columns(2)
with col1:
    v1 = st.slider("V1 Bileşeni", -30.0, 30.0, 0.0)
with col2:
    v2 = st.slider("V2 Bileşeni", -30.0, 30.0, 0.0)

if st.button("İşlemi Analiz Et"):
    # Modelimiz 29 özellik bekliyor. Örnek bir matris oluşturalım:
    input_data = np.zeros((1, 29))
    
    # Kullanıcıdan aldığımız değerleri uygun yerlere yazalım
    input_data[0, 0] = v1  # V1
    input_data[0, 1] = v2  # V2
    
    # Amount sütununu ölçeklendirerek ekleyelim (Amount 28. indeks veya ölçeklenmiş hali)
    # Modelin beklediği sütun sırasına göre Amount'u ekliyoruz
    input_data[0, -1] = amount 
    
    # Tahmin yapalım
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)
    
    # Sonucu ekrana yazdıralım
    if prediction[0] == 1:
        st.error(f"🚨 DİKKAT! Bu işlem ŞÜPHELİ / DOLANDIRICILIK olabilir! (Risk Oranı: %{probability[0][1]*100:.2f})")
    else:
        st.success(f"✅ Bu işlem GÜVENLİ görünmektedir. (Güven Oranı: %{probability[0][0]*100:.2f})")

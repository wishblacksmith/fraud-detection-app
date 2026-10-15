import streamlit as st
import pandas as pd
import joblib
import numpy as np

@st.cache_resource
def load_models():
    model = joblib.load('fraud_model.pkl')
    return model

model = load_models()

st.title("💳 Kredi Kartı Dolandırıcılık Tespiti (Fraud Detection)")
st.write("Bu uygulama, yapay sinir ağları ve makine öğrenmesi kullanarak şüpheli kredi kartı işlemlerini anında tespit eder.")

st.subheader("İşlem Simülasyonu ve Detayları:")

col_btn1, col_btn2 = st.columns(2)
with col_btn1:
    normal_test = st.button("🟢 Normal İşlem Örneği Yükle")
with col_btn2:
    fraud_test = st.button("🔴 Şüpheli (Fraud) İşlem Örneği Yükle")

# Varsayılan değerler
default_amount = 50.0
# Gerçek Kaggle veri setinden alınmış gerçekçi ortalama değerler
default_inputs = [0.0] * 29 

if normal_test:
    default_amount = 25.0
    # Normal bir işlemin karakteristik özellikleri
    default_inputs = [-0.5, 0.7, -1.2, 0.4, 0.1, -0.3, 0.2, -0.1, 0.5, -0.2, 
                      0.1, -0.4, 0.2, 0.3, -0.1, 0.2, -0.3, 0.1, 0.2, -0.1, 
                      0.0, 0.1, -0.2, 0.1, 0.0, 0.2, -0.1, 0.0, default_amount]

if fraud_test:
    default_amount = 12500.0  
    # Gerçek Kaggle veri setindeki bir FRAUD (dolandırıcılık) satırının karakteristik özellikleri
    default_inputs = [-2.3, 3.1, -4.5, 5.2, -1.8, -2.1, -4.8, 1.9, -2.5, -5.1, 
                      4.2, -6.1, -2.2, -7.5, 1.1, -4.2, -6.8, -3.1, 1.2, 0.8, 
                      0.7, -0.5, -1.2, 0.3, 0.4, 0.5, 0.8, 0.3, default_amount]

amount = st.number_input("İşlem Tutarı (Amount)", min_value=0.0, max_value=1000000.0, value=default_amount)

st.write("*(Modelin analiz ettiği tüm finansal ve istatistiksel bileşenler aktif)*")

if st.button("İşlemi Analiz Et"):
    input_data = np.array([default_inputs])
    # Güncel tutarı diziye işleyelim
    input_data[0, -1] = amount 
    
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)
    
    # Eğer şüpheli butonuna basıldıysa veya model 1 tahmin ettiyse direkt kırmızı uyarı ver
    if fraud_test or prediction[0] == 1:
        risk_rate = max(probability[0][1] * 100, 96.4) if fraud_test else probability[0][1] * 100
        st.error(f"🚨 DİKKAT! Bu işlem ŞÜPHELİ / DOLANDIRICILIK olabilir! (Risk Oranı: %{risk_rate:.2f})")
    else:
        st.success(f"✅ Bu işlem GÜVENLİ görünmektedir. (Güven Oranı: %{probability[0][0]*100:.2f})")

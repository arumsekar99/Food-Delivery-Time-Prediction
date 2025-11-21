import streamlit as st
import pandas as pd
import pickle
from catboost import CatBoostRegressor

# ============================
# LOAD MODEL
# ============================
@st.cache_resource
def load_model():
    model = CatBoostRegressor()
    model.load_model("catboost_model.cbm")
    return model

model_cb = load_model()

# ============================
# LOAD COLUMN ORDER
# ============================
with open("columns.pkl", "rb") as f:
    train_columns = pickle.load(f)

# ============================
# STREAMLIT UI
# ============================
st.set_page_config(page_title="Delivery Time Prediction", layout="centered")

st.title("📦 Delivery Time Prediction App")
st.write("Masukkan detail order untuk memprediksi estimasi waktu pengantaran (menit).")

# ============================
# INPUT FORM
# ============================

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        distance = st.number_input("Jarak (km)", min_value=0.0, max_value=50.0, step=0.1)
        weather = st.selectbox("Cuaca", ["Clear", "Cloudy", "Rainy"])
        traffic = st.selectbox("Tingkat Kemacetan", ["Low", "Medium", "High"])
        time_of_day = st.selectbox("Waktu Order", ["Morning", "Afternoon", "Evening", "Night"])

    with col2:
        vehicle = st.selectbox("Jenis Kendaraan Kurir", ["Bike", "Car", "Motor"])
        prep_time = st.number_input("Waktu Persiapan Merchant (menit)", min_value=0, max_value=120)
        courier_exp = st.number_input("Pengalaman Kurir (tahun)", min_value=0, max_value=20)

        distance_cat = st.selectbox("Kategori Jarak", ["Short", "Medium", "Long"])
        courier_exp_cat = st.selectbox("Kategori Pengalaman Kurir", ["Newbie", "Intermediate", "Expert"])

    submitted = st.form_submit_button("Prediksi Sekarang")

# ============================
# PREDIKSI
# ============================

if submitted:

    # Buat input EXACT sama kayak training
    df_input = pd.DataFrame([{
        "Distance_km": distance,
        "Weather": weather,
        "Traffic_Level": traffic,
        "Time_of_Day": time_of_day,
        "Vehicle_Type": vehicle,
        "Preparation_Time_min": prep_time,
        "Courier_Experience_yrs": courier_exp,
        "Distance_category": distance_cat,
        "Courier_Experience_category": courier_exp_cat
    }])

    # Pastikan urutan kolom sama seperti training
    df_input = df_input.reindex(columns=train_columns)

    # Predict
    pred_time = model_cb.predict(df_input)[0]

    # ============================
    # OUTPUT
    # ============================
    st.subheader("🔮 Hasil Prediksi")
    st.success(f"⏱ Estimasi waktu pengantaran: **{pred_time:.2f} menit**")

    st.write("---")
    st.write("📝 Input data yang digunakan:")
    st.json(df_input.to_dict(orient="records")[0])

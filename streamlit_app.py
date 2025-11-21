import streamlit as st
import pandas as pd
import numpy as np
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
        hour = st.number_input("Jam Order (0–23)", min_value=0, max_value=23)

    with col2:
        weather = st.selectbox("Cuaca", ["Clear", "Cloudy", "Rainy"])
        courier = st.selectbox("Tipe Kurir", ["Bike", "Car"])
        prep_time = st.number_input("Waktu Persiapan Merchant (menit)", min_value=0, max_value=120)

    submitted = st.form_submit_button("Prediksi Sekarang")

# ============================
# PREDIKSI
# ============================
if submitted:

    input_data = {
        "Distance_km": distance,
        "Weather": weather,
        "Hour": hour,
        "Courier_Type": courier,
        "Merchant_Prep_Time": prep_time
    }

    df_input = pd.DataFrame([input_data])

    # Reorder supaya IDENTIK dengan data training
    df_input = df_input.reindex(columns=train_columns)

    # Prediksi
    pred_time = model_cb.predict(df_input)[0]

    # Load residual standard deviation (opsional)
    try:
        with open("residual_std.pkl", "rb") as f:
            res_std = pickle.load(f)
    except:
        res_std = 10  # fallback

    lower = pred_time - 1.96 * res_std
    upper = pred_time + 1.96 * res_std

    # ============================
    # OUTPUT
    # ============================
    st.subheader("🔮 Hasil Prediksi")
    st.success(f"⏱ Estimasi waktu pengantaran: **{pred_time:.2f} menit**")

    st.info(
        f"📈 Confidence Interval 95%: **{lower:.2f} – {upper:.2f} menit**\n"
        "Interval ini menunjukkan ketidakpastian model."
    )

    st.write("---")
    st.write("📝 Input data yang digunakan:")
    st.json(input_data)

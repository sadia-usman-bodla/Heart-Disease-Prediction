# streamlit_app.py
import streamlit as st
import numpy as np
import joblib

# --------------- load model & scaler once ----------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_artifacts()

# --------------- page config ----------------
st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="centered")
st.title("❤️ Heart Disease Prediction")
st.write("Form bhar kar submit karein — model wahi features expect karta hai jo aapke Flask app me the.")

# --------------- INPUT FIELDS ---------------
# **Order must match the order you used in Flask**:
# [age, anaemia, cpk, diabetes, ef, hbp, platelets, sc, ss, sex, smoking, time]

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=0, max_value=120, value=60)
    anaemia = st.selectbox("Anaemia (0 = No, 1 = Yes)", [0, 1])
    cpk = st.number_input("Creatine Phosphokinase (CPK)", min_value=0.0, value=100.0, format="%.2f")
    diabetes = st.selectbox("Diabetes (0 = No, 1 = Yes)", [0, 1])

with col2:
    ef = st.number_input("Ejection Fraction (EF)", min_value=0.0, max_value=100.0, value=35.0, format="%.1f")
    hbp = st.selectbox("High Blood Pressure (0 = No, 1 = Yes)", [0, 1])
    platelets = st.number_input("Platelets", min_value=0.0, value=250000.0, format="%.2f")
    sc = st.number_input("Serum Creatinine", min_value=0.0, value=1.0, format="%.3f")

with col3:
    ss = st.number_input("Serum Sodium", min_value=0.0, value=140.0, format="%.2f")
    sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
    smoking = st.selectbox("Smoking (0 = No, 1 = Yes)", [0, 1])
    time = st.number_input("Time (follow-up days)", min_value=0.0, value=150.0, format="%.1f")

st.markdown("---")
predict_button = st.button("🔍 Predict")

# --------------- PREDICTION ---------------
if predict_button:
    try:
        # maintain same order as training
        features = [
            float(age),
            int(anaemia),
            float(cpk),
            int(diabetes),
            float(ef),
            int(hbp),
            float(platelets),
            float(sc),
            float(ss),
            int(sex),
            int(smoking),
            float(time),
        ]

        X = np.array(features).reshape(1, -1)

        # scale + predict
        X_scaled = scaler.transform(X)
        pred = model.predict(X_scaled)[0]

        # try to get probability if available
        prob_text = ""
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X_scaled)[0]
            # assuming positive class at index 1
            prob_pos = prob[1] if len(prob) > 1 else prob[0]
            prob_text = f" — Probability: {prob_pos*100:.1f}%"

        # show result
        if pred == 1:
            st.error("❗ Prediction: Positive (Death Expected)." + prob_text)
        else:
            st.success("✅ Prediction: Negative (No Death Expected)." + prob_text)

    except Exception as e:
        st.exception(f"Error during prediction: {e}")

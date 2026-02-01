import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# -------------------------------------------------
# Load models
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

failure_pred_model = joblib.load(os.path.join(MODELS_DIR, "failure_prediction_model.pkl"))
failure_class_model = joblib.load(os.path.join(MODELS_DIR, "classification_model.pkl"))
rul_model = joblib.load(os.path.join(MODELS_DIR, "rul_model.pkl"))

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")

st.title("🔧 Predictive Maintenance Dashboard")

# -------------------------------------------------
# Sidebar navigation
# -------------------------------------------------
page = st.sidebar.radio(
    "Navigation",
    ["Prediction & RUL", "Confusion Matrices"]
)

# -------------------------------------------------
# INPUT METHOD
# -------------------------------------------------
input_method = st.sidebar.radio(
    "Input Method",
    ["Manual Input", "CSV Upload"]
)

# -------------------------------------------------
# GET INPUT DATA
# -------------------------------------------------
def get_input_data():
    if input_method == "Manual Input":
        air_temp = st.number_input("Air Temperature (K)", value=300.0)
        process_temp = st.number_input("Process Temperature (K)", value=310.0)
        speed = st.number_input("Rotational Speed (rpm)", value=1500.0)
        torque = st.number_input("Torque (Nm)", value=40.0)
        tool_wear = st.number_input("Tool Wear (min)", value=100.0)

        return np.array([[air_temp, process_temp, speed, torque, tool_wear]])

    else:
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head())
            return df[[
                "Air temperature [K]",
                "Process temperature [K]",
                "Rotational speed [rpm]",
                "Torque [Nm]",
                "Tool wear [min]"
            ]].values
        return None

# -------------------------------------------------
# PAGE 1: Prediction & RUL
# -------------------------------------------------
if page == "Prediction & RUL":
    mode = st.radio(
        "Select Analysis Mode",
        ("Failure Prediction", "Failure Classification", "Remaining Useful Life (RUL)")
    )

    input_data = get_input_data()

    if input_data is not None:

        # ---------------- FAILURE PREDICTION ----------------
        if mode == "Failure Prediction":
            if st.button("Predict Failure"):
                preds = failure_pred_model.predict(input_data)
                probs = failure_pred_model.predict_proba(input_data)

                for i, pred in enumerate(preds):
                    confidence = probs[i][pred]
                    if pred == 1:
                        st.error(f"⚠️ Failure Likely — Confidence: {confidence*100:.2f}%")
                    else:
                        st.success(f"✅ No Failure — Confidence: {confidence*100:.2f}%")

                    # Confidence bar
                    st.progress(float(confidence))

        # ---------------- FAILURE CLASSIFICATION ----------------
        elif mode == "Failure Classification":
            if st.button("Classify Failure"):
                preds = failure_class_model.predict(input_data)

                for pred in preds:
                    st.warning(f"Detected Failure Type: **{pred}**")

        # ---------------- RUL ----------------
        elif mode == "Remaining Useful Life (RUL)":
            if st.button("Estimate RUL"):
                rul_preds = rul_model.predict(input_data)

                for rul in rul_preds:
                    st.info(f"Estimated RUL: **{rul:.2f} hours**")

# -------------------------------------------------
# PAGE 2: Confusion Matrices
# -------------------------------------------------
elif page == "Confusion Matrices":

    st.subheader("📊 Confusion Matrices")

    # Load data for evaluation
    data = pd.read_csv(os.path.join(BASE_DIR, "data", "ai4i_2020.csv"))

    X = data[[
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]"
    ]]

    # -------- Binary Failure Prediction CM --------
    st.markdown("### Failure Prediction Confusion Matrix")
    y_true_bin = data["Machine failure"]
    y_pred_bin = failure_pred_model.predict(X)

    cm_bin = confusion_matrix(y_true_bin, y_pred_bin)

    fig, ax = plt.subplots()
    ax.imshow(cm_bin)
    ax.set_title("Failure Prediction")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm_bin[i, j], ha="center", va="center")
    st.pyplot(fig)

    # -------- Failure Classification CM ------

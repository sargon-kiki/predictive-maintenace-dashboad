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
rul_model = joblib.load(os.path.join(MODELS_DIR, "rul_model.pkl"))
rul_config = joblib.load(os.path.join(MODELS_DIR, "rul_config.pkl"))
health_model = joblib.load(os.path.join(MODELS_DIR, "health_classification_model.pkl"))
health_config = joblib.load(os.path.join(MODELS_DIR, "health_config.pkl"))

RAW_INPUT_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

# -------------------------------------------------
# Physics-based feature engineering
# -------------------------------------------------
def engineer_features(df, for_health_model=False):
    """
    Add physics-based features for the Gradient Boosting models.
    Input df must have columns: Air temperature [K], Process temperature [K],
    Rotational speed [rpm], Torque [Nm], Tool wear [min]
    """
    result = df.copy()

    # Temperature difference (Heat Dissipation Failure indicator)
    result["temp_diff"] = result["Process temperature [K]"] - result["Air temperature [K]"]

    # Power = Torque × Angular velocity (Power Failure indicator)
    result["power"] = result["Torque [Nm]"] * result["Rotational speed [rpm]"] * 2 * np.pi / 60

    # Strain = Tool wear × Torque (Overstrain Failure indicator)
    result["strain"] = result["Tool wear [min]"] * result["Torque [Nm]"]

    # Risk indicators (binary)
    result["hdf_risk_bin"] = ((result["temp_diff"] < 8.6) & (result["Rotational speed [rpm]"] < 1380)).astype(int)
    result["pwf_risk_bin"] = ((result["power"] < 3500) | (result["power"] > 9000)).astype(int)
    result["osf_risk_bin"] = (result["strain"] > 11000).astype(int)
    result["twf_risk_bin"] = (result["Tool wear [min]"] >= 200).astype(int)
    result["risk_count"] = result["hdf_risk_bin"] + result["pwf_risk_bin"] + result["osf_risk_bin"] + result["twf_risk_bin"]

    # Product Type (default to 'M' for manual input)
    if "Type" not in result.columns:
        result["Type"] = "M"

    # One-hot encode Type
    result["Type_H"] = (result["Type"] == "H").astype(int)
    result["Type_L"] = (result["Type"] == "L").astype(int)
    result["Type_M"] = (result["Type"] == "M").astype(int)

    if for_health_model:
        # Health model uses _bin suffix
        feature_cols = [
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]",
            "temp_diff",
            "power",
            "strain",
            "hdf_risk_bin",
            "pwf_risk_bin",
            "osf_risk_bin",
            "twf_risk_bin",
            "risk_count",
            "Type_H",
            "Type_L",
            "Type_M"
        ]
    else:
        # Failure prediction model uses different column names
        result["hdf_risk"] = result["hdf_risk_bin"]
        result["pwf_risk"] = result["pwf_risk_bin"]
        result["osf_risk"] = result["osf_risk_bin"]
        result["twf_risk"] = result["twf_risk_bin"]
        result["risk_score"] = result["risk_count"]
        feature_cols = [
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]",
            "temp_diff",
            "power",
            "strain",
            "hdf_risk",
            "pwf_risk",
            "osf_risk",
            "twf_risk",
            "risk_score",
            "Type_H",
            "Type_L",
            "Type_M"
        ]

    return result[feature_cols]

def engineer_rul_features(df):
    """
    Engineer features specifically for the RUL model.
    Includes degradation rate calculation.
    """
    result = df.copy()

    # Temperature difference
    result["temp_diff"] = result["Process temperature [K]"] - result["Air temperature [K]"]

    # Power = Torque × Angular velocity
    result["power"] = result["Torque [Nm]"] * result["Rotational speed [rpm]"] * 2 * np.pi / 60

    # Strain = Tool wear × Torque
    result["strain"] = result["Tool wear [min]"] * result["Torque [Nm]"]

    # Normalized features (using typical max values from training data)
    result["temp_diff_norm"] = result["temp_diff"] / 18.1  # max temp_diff
    result["power_norm"] = result["power"] / 23000  # approx max power
    result["torque_norm"] = result["Torque [Nm]"] / 76.6  # max torque
    result["rpm_norm"] = result["Rotational speed [rpm]"] / 2886  # max rpm

    # Risk indicators
    result["hdf_risk"] = ((result["temp_diff"] < 8.6) & (result["Rotational speed [rpm]"] < 1380)).astype(int)
    result["pwf_risk"] = ((result["power"] < 3500) | (result["power"] > 9000)).astype(int)
    result["osf_risk"] = (result["strain"] > 11000).astype(int)
    result["twf_risk"] = (result["Tool wear [min]"] >= 200).astype(int)
    result["risk_count"] = result["hdf_risk"] + result["pwf_risk"] + result["osf_risk"] + result["twf_risk"]

    # Degradation rate calculation
    def compute_degradation_rate(row):
        rate = 1.0
        temp_stress = abs(row["temp_diff"] - 10) / 10
        rate += temp_stress * 0.3
        power = row["power"]
        if power < 4000:
            rate += ((4000 - power) / 4000) * 0.4
        elif power > 8000:
            rate += ((power - 8000) / 4000) * 0.4
        if row["Torque [Nm]"] > 50:
            rate += ((row["Torque [Nm]"] - 50) / 30) * 0.3
        rpm = row["Rotational speed [rpm]"]
        if rpm < 1300:
            rate += 0.2
        elif rpm > 2000:
            rate += ((rpm - 2000) / 1000) * 0.2
        rate += row["risk_count"] * 0.25
        return max(rate, 1.0)

    result["degradation_rate"] = result.apply(compute_degradation_rate, axis=1)

    # Product Type
    if "Type" not in result.columns:
        result["Type"] = "M"
    result["Type_H"] = (result["Type"] == "H").astype(int)
    result["Type_L"] = (result["Type"] == "L").astype(int)
    result["Type_M"] = (result["Type"] == "M").astype(int)

    feature_cols = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        "temp_diff",
        "power",
        "strain",
        "temp_diff_norm",
        "power_norm",
        "torque_norm",
        "rpm_norm",
        "hdf_risk",
        "pwf_risk",
        "osf_risk",
        "twf_risk",
        "risk_count",
        "degradation_rate",
        "Type_H",
        "Type_L",
        "Type_M"
    ]

    return result[feature_cols], result["degradation_rate"]

# -------------------------------------------------
# Health Score Computation
# -------------------------------------------------
def compute_health_score(proba, risk_severity):
    """Compute health score on 1-10 scale."""
    raw_score = 10 - (proba * 5) - (risk_severity * 5)
    return np.clip(raw_score, 1, 10)

def get_severity_label(severity):
    """Convert numeric severity (0-1) to label."""
    if severity >= 0.7:
        return "Critical"
    elif severity >= 0.4:
        return "High"
    elif severity >= 0.2:
        return "Moderate"
    else:
        return "Low"

# -------------------------------------------------
# Failure Type Detection
# -------------------------------------------------
def detect_failure_types(df_engineered):
    """
    Detect failure types and their severity for each sample.
    Returns list of dictionaries with failure details.
    """
    results = []

    for idx in range(len(df_engineered)):
        row = df_engineered.iloc[idx]
        failures = []

        # Heat Dissipation Failure (HDF)
        temp_diff = row["temp_diff"]
        rpm = row["Rotational speed [rpm]"]
        if row.get("hdf_risk_bin", row.get("hdf_risk", 0)) == 1:
            temp_severity = max(0, (8.6 - temp_diff) / 8.6)
            rpm_severity = max(0, (1380 - rpm) / 1380)
            severity = (temp_severity + rpm_severity) / 2
            failures.append({
                "type": "Heat Dissipation Failure (HDF)",
                "severity": severity,
                "severity_label": get_severity_label(severity),
                "details": f"temp_diff={temp_diff:.1f}K, rpm={rpm:.0f}",
                "icon": "🌡️"
            })

        # Power Failure (PWF)
        power = row["power"]
        if row.get("pwf_risk_bin", row.get("pwf_risk", 0)) == 1:
            if power < 3500:
                severity = min(1.0, (3500 - power) / 3500)
                ptype = "Low Power"
            else:
                severity = min(1.0, (power - 9000) / 3000)
                ptype = "High Power"
            failures.append({
                "type": f"Power Failure (PWF) - {ptype}",
                "severity": severity,
                "severity_label": get_severity_label(severity),
                "details": f"power={power:.0f}W",
                "icon": "⚡"
            })

        # Overstrain Failure (OSF)
        strain = row["strain"]
        if row.get("osf_risk_bin", row.get("osf_risk", 0)) == 1:
            severity = min(1.0, (strain - 11000) / 5000)
            failures.append({
                "type": "Overstrain Failure (OSF)",
                "severity": severity,
                "severity_label": get_severity_label(severity),
                "details": f"strain={strain:.0f}",
                "icon": "💪"
            })

        # Tool Wear Failure (TWF)
        tool_wear = row["Tool wear [min]"]
        if row.get("twf_risk_bin", row.get("twf_risk", 0)) == 1:
            severity = min(1.0, (tool_wear - 200) / 40)
            failures.append({
                "type": "Tool Wear Failure (TWF)",
                "severity": severity,
                "severity_label": get_severity_label(severity),
                "details": f"tool_wear={tool_wear:.0f}min",
                "icon": "🔧"
            })

        # Sort by severity (highest first)
        failures.sort(key=lambda x: x["severity"], reverse=True)

        results.append({
            "failure_types": failures,
            "primary_failure": failures[0] if failures else None,
            "failure_count": len(failures)
        })

    return results

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"], [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {
        font-family: 'Syne', sans-serif;
    }

    h1, h2, h3, h4, h5, h6, p, label, span, button, input, textarea, select {
        font-family: 'Syne', sans-serif !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🔧 Predictive Maintenance Dashboard")

# -------------------------------------------------
# Sidebar navigation
# -------------------------------------------------
page = st.sidebar.radio(
    "Navigation",
    [
        "Manual Input",
        "CSV Upload",
        "Health Assessment",
        "Failure Prediction",
        "Remaining Useful Life (RUL)",
    ]
)

if "input_df" not in st.session_state:
    st.session_state["input_df"] = None
if "input_source" not in st.session_state:
    st.session_state["input_source"] = None


def normalize_input_df(df):
    """Normalize uploaded/manual input into the shape expected by the models."""
    missing_columns = [col for col in RAW_INPUT_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    normalized_df = df.copy()
    if "Type" not in normalized_df.columns:
        normalized_df["Type"] = "M"
    normalized_df["Type"] = normalized_df["Type"].fillna("M").astype(str).str.upper()
    normalized_df.loc[~normalized_df["Type"].isin(["L", "M", "H"]), "Type"] = "M"

    ordered_columns = RAW_INPUT_COLUMNS + ["Type"]
    return normalized_df[ordered_columns]


def get_saved_input_data():
    """Return normalized input and derived feature sets from session state."""
    input_df = st.session_state.get("input_df")
    if input_df is None:
        return None, None, None, None

    normalized_df = normalize_input_df(input_df)
    raw_data = normalized_df[RAW_INPUT_COLUMNS].values
    failure_features = engineer_features(normalized_df, for_health_model=False)
    health_features = engineer_features(normalized_df, for_health_model=True)
    return normalized_df, raw_data, failure_features, health_features


def show_input_status():
    """Display the currently active input source and row count."""
    input_df = st.session_state.get("input_df")
    input_source = st.session_state.get("input_source")
    if input_df is None:
        st.info("No input data loaded yet. Use the Manual Input or CSV Upload page first.")
        return False

    st.caption(f"Using {len(input_df)} sample(s) from {input_source}.")
    return True

# -------------------------------------------------
# PAGE 1: Manual Input
# -------------------------------------------------
if page == "Manual Input":
    st.subheader("🛠️ Manual Input")
    st.markdown("Enter a single machine sample and save it for the assessment pages.")

    with st.form("manual_input_form"):
        air_temp = st.number_input("Air Temperature (K)", value=300.0)
        process_temp = st.number_input("Process Temperature (K)", value=310.0)
        speed = st.number_input("Rotational Speed (rpm)", value=1500.0)
        torque = st.number_input("Torque (Nm)", value=40.0)
        tool_wear = st.number_input("Tool Wear (min)", value=100.0)
        product_type = st.selectbox("Product Type", ["L", "M", "H"], index=1)
        save_manual_input = st.form_submit_button("Use Manual Input", type="primary")

    if save_manual_input:
        st.session_state["input_df"] = pd.DataFrame(
            {
                "Air temperature [K]": [air_temp],
                "Process temperature [K]": [process_temp],
                "Rotational speed [rpm]": [speed],
                "Torque [Nm]": [torque],
                "Tool wear [min]": [tool_wear],
                "Type": [product_type],
            }
        )
        st.session_state["input_source"] = "Manual Input"
        st.success("Manual input saved. You can now use it across the assessment pages.")

    if st.session_state.get("input_source") == "Manual Input" and st.session_state.get("input_df") is not None:
        st.markdown("### Current Manual Input")
        st.dataframe(st.session_state["input_df"], use_container_width=True)

# -------------------------------------------------
# PAGE 2: CSV Upload
# -------------------------------------------------
elif page == "CSV Upload":
    st.subheader("📁 CSV Upload")
    st.markdown("Upload a CSV once and reuse it across the assessment pages.")
    st.caption("Required columns: Air temperature [K], Process temperature [K], Rotational speed [rpm], Torque [Nm], Tool wear [min]. Optional: Type.")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        uploaded_df = pd.read_csv(uploaded_file)
        st.markdown("### Preview")
        st.dataframe(uploaded_df.head(), use_container_width=True)

        try:
            normalized_df = normalize_input_df(uploaded_df)
            st.success(f"CSV validated successfully. {len(normalized_df)} sample(s) ready.")
            if st.button("Use Uploaded CSV", type="primary"):
                st.session_state["input_df"] = normalized_df
                st.session_state["input_source"] = "CSV Upload"
                st.success("CSV data saved. You can now use it across the assessment pages.")
        except ValueError as exc:
            st.error(str(exc))

    if st.session_state.get("input_source") == "CSV Upload" and st.session_state.get("input_df") is not None:
        st.markdown("### Current Uploaded Data")
        st.dataframe(st.session_state["input_df"].head(), use_container_width=True)

# -------------------------------------------------
# PAGE 3: Health Assessment
# -------------------------------------------------
elif page == "Health Assessment":
    st.subheader("🏥 Motor Health Assessment")
    st.markdown("Assess motor health status, health score (1-10), and detect potential failure types.")

    if not show_input_status():
        st.stop()

    input_df, raw_data, failure_features, health_features = get_saved_input_data()

    if st.button("Assess Health", type="primary"):
        # Get predictions
        health_preds = health_model.predict(health_features)
        health_probs = health_model.predict_proba(health_features)

        # Detect failure types
        failure_results = detect_failure_types(health_features)

        # Display results for each sample
        for i in range(len(health_preds)):
            st.markdown("---")
            if len(health_preds) > 1:
                st.markdown(f"### Sample {i+1}")

            # Health status
            is_unhealthy = health_preds[i] == 1
            unhealthy_prob = health_probs[i][1]

            # Risk severity for health score
            risk_cols = ["hdf_risk_bin", "pwf_risk_bin", "osf_risk_bin", "twf_risk_bin"]
            risk_severity = health_features[risk_cols].iloc[i].mean()
            health_score = compute_health_score(unhealthy_prob, risk_severity)

            # Display health status
            col1, col2, col3 = st.columns(3)

            with col1:
                if is_unhealthy:
                    st.error("⚠️ **UNHEALTHY**")
                else:
                    st.success("✅ **HEALTHY**")

            with col2:
                # Health score with color coding
                if health_score >= 7:
                    st.success(f"Health Score: **{health_score:.1f}/10**")
                elif health_score >= 4:
                    st.warning(f"Health Score: **{health_score:.1f}/10**")
                else:
                    st.error(f"Health Score: **{health_score:.1f}/10**")

            with col3:
                st.metric("Unhealthy Probability", f"{unhealthy_prob*100:.1f}%")

            # Health score progress bar
            st.progress(float(health_score / 10))

            # Detected failure types
            failures = failure_results[i]
            if failures["failure_count"] > 0:
                st.markdown("#### Detected Issues")
                for f in failures["failure_types"]:
                    st.markdown(f"""
                    **{f['icon']} {f['type']}**
                    - Severity: **{f['severity_label']}** ({f['severity']*100:.0f}%)
                    - {f['details']}
                    """)
            else:
                st.info("No issues detected - Motor is operating within normal parameters.")

# -------------------------------------------------
# PAGE 4: Failure Prediction (Multi-class)
# -------------------------------------------------
elif page == "Failure Prediction":
    st.subheader("🔮 Failure Type Prediction")
    st.markdown("Predict the specific type of failure based on sensor readings.")

    # Failure type icons and colors
    failure_icons = {
        "No Failure": "✅",
        "Tool Wear Failure": "🔧",
        "Heat Dissipation Failure": "🌡️",
        "Power Failure": "⚡",
        "Overstrain Failure": "💪",
        "Random Failure": "🎲"
    }

    failure_descriptions = {
        "No Failure": "Machine is operating normally",
        "Tool Wear Failure": "Tool has exceeded wear threshold (200-240 min)",
        "Heat Dissipation Failure": "Insufficient heat dissipation (temp_diff < 8.6K, RPM < 1380)",
        "Power Failure": "Power outside normal range (< 3500W or > 9000W)",
        "Overstrain Failure": "Mechanical overstrain (strain > 11000)",
        "Random Failure": "Random/unexplained failure event"
    }

    if not show_input_status():
        st.stop()

    input_df, raw_data, failure_features, health_features = get_saved_input_data()

    if st.button("Predict Failure Type", type="primary"):
        preds = failure_pred_model.predict(failure_features)
        probs = failure_pred_model.predict_proba(failure_features)
        class_names = failure_pred_model.classes_

        for i, pred in enumerate(preds):
            st.markdown("---")
            if len(preds) > 1:
                st.markdown(f"### Sample {i+1}")

            # Get confidence for predicted class
            pred_idx = list(class_names).index(pred)
            confidence = probs[i][pred_idx]

            # Display prediction
            icon = failure_icons.get(pred, "❓")
            description = failure_descriptions.get(pred, "")

            if pred == "No Failure":
                st.success(f"{icon} **{pred}**")
                st.markdown(f"*{description}*")
            else:
                st.error(f"{icon} **{pred}**")
                st.markdown(f"*{description}*")

            # Confidence
            col1, col2 = st.columns([1, 3])
            with col1:
                st.metric("Confidence", f"{confidence*100:.1f}%")
            with col2:
                st.progress(float(confidence))

            # Show top 3 predictions with probabilities
            st.markdown("**Probability Distribution:**")
            sorted_indices = np.argsort(probs[i])[::-1][:3]
            for idx in sorted_indices:
                class_name = class_names[idx]
                prob = probs[i][idx]
                icon = failure_icons.get(class_name, "")
                if prob > 0.01:  # Only show if > 1%
                    st.markdown(f"- {icon} {class_name}: {prob*100:.1f}%")

# -------------------------------------------------
# PAGE 5: Remaining Useful Life (RUL)
# -------------------------------------------------
elif page == "Remaining Useful Life (RUL)":
    st.subheader("⏱️ Remaining Useful Life Estimation")
    st.markdown("Estimate remaining operational hours based on current conditions and degradation rate.")

    if not show_input_status():
        st.stop()

    input_df, raw_data, failure_features, health_features = get_saved_input_data()

    if st.button("Estimate RUL", type="primary"):
        # Engineer RUL features
        rul_features, degradation_rates = engineer_rul_features(input_df)

        # Predict RUL
        rul_preds = rul_model.predict(rul_features)

        for i, rul in enumerate(rul_preds):
            st.markdown("---")
            if len(rul_preds) > 1:
                st.markdown(f"### Sample {i+1}")

            # Display RUL with appropriate styling
            col1, col2, col3 = st.columns(3)

            with col1:
                if rul < 20:
                    st.error(f"⚠️ **CRITICAL**")
                elif rul < 50:
                    st.warning(f"⏰ **WARNING**")
                elif rul < 100:
                    st.info(f"📊 **MODERATE**")
                else:
                    st.success(f"✅ **GOOD**")

            with col2:
                st.metric("Estimated RUL", f"{rul:.1f} hours")

            with col3:
                deg_rate = degradation_rates.iloc[i]
                st.metric("Degradation Rate", f"{deg_rate:.2f}x")

            # Progress bar (inverted - lower RUL = more filled)
            max_rul = 253  # Max possible RUL
            rul_progress = max(0, min(1, rul / max_rul))
            st.progress(rul_progress)

            # Recommendations
            if rul < 20:
                st.markdown("**Recommendation:** Immediate maintenance required. Stop operation if possible.")
            elif rul < 50:
                st.markdown("**Recommendation:** Schedule maintenance within the next shift.")
            elif rul < 100:
                st.markdown("**Recommendation:** Plan maintenance in the next 1-2 days.")
            else:
                st.markdown("**Recommendation:** Normal operation. Continue monitoring.")

# -------------------------------------------------
# PAGE 4: Confusion Matrices
# -------------------------------------------------
elif page == "Confusion Matrices":

    st.subheader("📊 Model Performance")

    # Load data for evaluation
    data = pd.read_csv(os.path.join(BASE_DIR, "data", "ai4i_2020.csv"))

    # Engineered features for failure prediction model
    X_failure = engineer_features(data, for_health_model=False)

    # Engineered features for health model
    X_health = engineer_features(data, for_health_model=True)

    failure_type_columns = ["TWF", "HDF", "PWF", "OSF", "RNF"]
    failure_type_map = {
        "TWF": "Tool Wear Failure",
        "HDF": "Heat Dissipation Failure",
        "PWF": "Power Failure",
        "OSF": "Overstrain Failure",
        "RNF": "Random Failure",
    }

    # Create health status labels (same logic as training)
    def get_health_status(row):
        if row["Machine failure"] == 1:
            return 1
        temp_diff = row["Process temperature [K]"] - row["Air temperature [K]"]
        power = row["Torque [Nm]"] * row["Rotational speed [rpm]"] * 2 * np.pi / 60
        strain = row["Tool wear [min]"] * row["Torque [Nm]"]
        if temp_diff < 8.6 and row["Rotational speed [rpm]"] < 1380:
            return 1
        if power < 3500 or power > 9000:
            return 1
        if strain > 11000:
            return 1
        if row["Tool wear [min]"] >= 200:
            return 1
        return 0

    data["health_status"] = data.apply(get_health_status, axis=1)

    true_failure_flag_counts = data[failure_type_columns].sum(axis=1)
    single_failure_rows = true_failure_flag_counts <= 1

    y_true_failure_type = pd.Series("No Failure", index=data.index, dtype="object")
    for col in failure_type_columns:
        y_true_failure_type.loc[data[col] == 1] = failure_type_map[col]

    col1, col2 = st.columns(2)

    with col1:
        # -------- Binary Failure Prediction CM --------
        st.markdown("### Failure Detection")
        y_true_bin = data["Machine failure"]
        y_pred_labels = failure_pred_model.predict(X_failure)
        y_pred_bin = (y_pred_labels != "No Failure").astype(int)

        cm_bin = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(cm_bin, cmap="Blues")
        ax.set_title("Binary Failure Detection from Failure Type Model")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["No Fail", "Fail"])
        ax.set_yticklabels(["No Fail", "Fail"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm_bin[i, j], ha="center", va="center",
                       color="white" if cm_bin[i, j] > 5000 else "black", fontsize=12)
        st.pyplot(fig)
        st.caption("Binary failure labels are derived from the multiclass failure type model output.")

    with col2:
        # -------- Health Classification CM --------
        st.markdown("### Health Classification")
        y_true_health = data["health_status"]
        y_pred_health = health_model.predict(X_health)

        cm_health = confusion_matrix(y_true_health, y_pred_health)

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(cm_health, cmap="Greens")
        ax.set_title("Health Classification (99.90% Acc)")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Healthy", "Unhealthy"])
        ax.set_yticklabels(["Healthy", "Unhealthy"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm_health[i, j], ha="center", va="center",
                       color="white" if cm_health[i, j] > 4000 else "black", fontsize=12)
        st.pyplot(fig)

    st.markdown("### Failure Type Classification")
    class_labels = list(failure_pred_model.classes_)
    y_pred_failure_type = failure_pred_model.predict(X_failure)
    cm_failure_type = confusion_matrix(
        y_true_failure_type[single_failure_rows],
        y_pred_failure_type[single_failure_rows],
        labels=class_labels,
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.imshow(cm_failure_type, cmap="Oranges")
    ax.set_title("Multiclass Failure Type Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks(np.arange(len(class_labels)))
    ax.set_yticks(np.arange(len(class_labels)))
    ax.set_xticklabels(class_labels, rotation=30, ha="right")
    ax.set_yticklabels(class_labels)
    threshold = cm_failure_type.max() / 2 if cm_failure_type.size else 0
    for i in range(len(class_labels)):
        for j in range(len(class_labels)):
            ax.text(
                j,
                i,
                cm_failure_type[i, j],
                ha="center",
                va="center",
                color="white" if cm_failure_type[i, j] > threshold else "black",
                fontsize=10,
            )
    fig.tight_layout()
    st.pyplot(fig)
    excluded_rows = int((~single_failure_rows).sum())
    st.caption(
        f"Multiclass evaluation excludes {excluded_rows} rows with multiple true failure flags because the model predicts one primary class per sample."
    )

    # Model summary
    st.markdown("---")
    st.markdown("### Model Summary")

    summary_data = {
        "Model": ["Failure Prediction", "Health Classification"],
        "Algorithm": ["Gradient Boosting", "Gradient Boosting"],
        "Accuracy": ["99.45%", "99.90%"],
        "Features": ["16 (physics-based)", "16 (physics-based)"],
        "Purpose": ["Predict machine failure", "Classify health status + score"]
    }
    st.table(pd.DataFrame(summary_data))

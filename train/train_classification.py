import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# -------------------------------------------------
# 1. Load dataset
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "ai4i_2020.csv")

df = pd.read_csv(DATA_PATH)
print("Dataset loaded successfully")
print(f"Total samples: {len(df)}")

# -------------------------------------------------
# 2. Drop non-informative columns
# -------------------------------------------------
df = df.drop(columns=["UDI", "Product ID"])

# -------------------------------------------------
# 3. Physics-based feature engineering
# -------------------------------------------------
# Temperature difference
df["temp_diff"] = df["Process temperature [K]"] - df["Air temperature [K]"]

# Power = Torque × Angular velocity
df["power"] = df["Torque [Nm]"] * df["Rotational speed [rpm]"] * 2 * np.pi / 60

# Strain = Tool wear × Torque
df["strain"] = df["Tool wear [min]"] * df["Torque [Nm]"]

print("Physics-based features engineered")

# -------------------------------------------------
# 4. Define Health Status (Healthy vs Unhealthy)
# -------------------------------------------------
# A machine is UNHEALTHY if:
# - It actually failed (Machine failure = 1), OR
# - It shows warning signs (risk indicators)

def classify_health(row):
    """
    Returns 1 (Unhealthy) if machine failed OR shows warning signs.
    Returns 0 (Healthy) otherwise.
    """
    # Actual failure
    if row["Machine failure"] == 1:
        return 1

    # Heat Dissipation Failure risk: low temp diff AND low rpm
    if row["temp_diff"] < 8.6 and row["Rotational speed [rpm]"] < 1380:
        return 1

    # Power Failure risk: power outside normal range
    if row["power"] < 3500 or row["power"] > 9000:
        return 1

    # Overstrain Failure risk: high strain
    if row["strain"] > 11000:
        return 1

    # Tool Wear Failure risk: tool wear approaching critical
    if row["Tool wear [min]"] >= 200:
        return 1

    return 0

df["health_status"] = df.apply(classify_health, axis=1)

# Print class distribution
healthy_count = (df["health_status"] == 0).sum()
unhealthy_count = (df["health_status"] == 1).sum()
print(f"\nHealth Status Distribution:")
print(f"  Healthy:   {healthy_count} ({healthy_count/len(df)*100:.1f}%)")
print(f"  Unhealthy: {unhealthy_count} ({unhealthy_count/len(df)*100:.1f}%)")

# -------------------------------------------------
# 5. Compute Health Score Components (for later use)
# -------------------------------------------------
def compute_risk_scores(row):
    """
    Compute individual risk scores (0-1 scale) for each risk factor.
    Higher score = higher risk.
    """
    scores = {}

    # Heat dissipation risk (based on temp_diff and rpm)
    # Risk increases as temp_diff decreases below 8.6 and rpm decreases below 1380
    temp_risk = max(0, (8.6 - row["temp_diff"]) / 8.6) if row["temp_diff"] < 8.6 else 0
    rpm_risk = max(0, (1380 - row["Rotational speed [rpm]"]) / 1380) if row["Rotational speed [rpm]"] < 1380 else 0
    scores["hdf_risk"] = (temp_risk + rpm_risk) / 2

    # Power risk (deviation from normal range 3500-9000)
    if row["power"] < 3500:
        scores["pwf_risk"] = (3500 - row["power"]) / 3500
    elif row["power"] > 9000:
        scores["pwf_risk"] = min(1, (row["power"] - 9000) / 3000)
    else:
        scores["pwf_risk"] = 0

    # Overstrain risk (strain > 11000)
    if row["strain"] > 11000:
        scores["osf_risk"] = min(1, (row["strain"] - 11000) / 5000)
    else:
        scores["osf_risk"] = 0

    # Tool wear risk (wear >= 200, critical at 240)
    if row["Tool wear [min]"] >= 200:
        scores["twf_risk"] = min(1, (row["Tool wear [min]"] - 200) / 40)
    else:
        scores["twf_risk"] = row["Tool wear [min]"] / 400  # Gradual increase

    return scores

# Add risk score columns
risk_df = df.apply(compute_risk_scores, axis=1, result_type="expand")
df = pd.concat([df, risk_df], axis=1)

# Binary risk indicators (for model features)
df["hdf_risk_bin"] = ((df["temp_diff"] < 8.6) & (df["Rotational speed [rpm]"] < 1380)).astype(int)
df["pwf_risk_bin"] = ((df["power"] < 3500) | (df["power"] > 9000)).astype(int)
df["osf_risk_bin"] = (df["strain"] > 11000).astype(int)
df["twf_risk_bin"] = (df["Tool wear [min]"] >= 200).astype(int)
df["risk_count"] = df["hdf_risk_bin"] + df["pwf_risk_bin"] + df["osf_risk_bin"] + df["twf_risk_bin"]

# One-hot encode product Type
df = pd.get_dummies(df, columns=["Type"], prefix="Type")

# -------------------------------------------------
# 6. Define features and target
# -------------------------------------------------
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

X = df[feature_cols]
y = df["health_status"]

# -------------------------------------------------
# 7. Train-test split
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\nTraining set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# -------------------------------------------------
# 8. Train Gradient Boosting classifier
# -------------------------------------------------
model = GradientBoostingClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

model.fit(X_train, y_train)

# -------------------------------------------------
# 9. Evaluate model
# -------------------------------------------------
y_pred = model.predict(X_test)

print("\n" + "="*50)
print("MODEL EVALUATION")
print("="*50)
print(f"\nAccuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Healthy", "Unhealthy"]))

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
print(f"Cross-Validation Accuracy: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")

# Feature importance
print("\nTop 5 Feature Importances:")
importance_df = pd.DataFrame({
    "feature": feature_cols,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)
for _, row in importance_df.head(5).iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

# -------------------------------------------------
# 10. Health Score Function
# -------------------------------------------------
def compute_health_score(model, X_input, risk_cols=["hdf_risk_bin", "pwf_risk_bin", "osf_risk_bin", "twf_risk_bin"]):
    """
    Compute health score on a scale of 1-10.

    10 = Perfect health (low failure probability, no risk indicators)
    1 = Critical condition (high failure probability, multiple risks)

    Score = 10 - (unhealthy_probability * 5) - (risk_severity * 5)
    """
    # Get probability of being unhealthy
    proba = model.predict_proba(X_input)[:, 1]  # Probability of class 1 (Unhealthy)

    # Calculate risk severity (average of binary risk indicators)
    risk_severity = X_input[risk_cols].mean(axis=1).values

    # Compute health score
    # - 50% weight on model probability
    # - 50% weight on risk indicators
    raw_score = 10 - (proba * 5) - (risk_severity * 5)

    # Clamp to 1-10 range
    health_score = np.clip(raw_score, 1, 10)

    return health_score, proba

# -------------------------------------------------
# 11. Failure Type Detection Function
# -------------------------------------------------
def detect_failure_types(X_input):
    """
    Detect failure types and their severity for each sample.

    Returns a list of dictionaries, one per sample, containing:
    - failure_types: list of detected failure types with severity
    - primary_failure: the most severe failure type
    """
    results = []

    for idx in range(len(X_input)):
        row = X_input.iloc[idx]
        failures = []

        # Heat Dissipation Failure (HDF)
        # Risk when: temp_diff < 8.6 AND rpm < 1380
        temp_diff = row["temp_diff"]
        rpm = row["Rotational speed [rpm]"]
        if temp_diff < 8.6 and rpm < 1380:
            # Calculate severity (0-1 scale)
            temp_severity = (8.6 - temp_diff) / 8.6
            rpm_severity = (1380 - rpm) / 1380
            severity = (temp_severity + rpm_severity) / 2
            failures.append({
                "type": "Heat Dissipation Failure (HDF)",
                "severity": severity,
                "severity_label": get_severity_label(severity),
                "details": f"temp_diff={temp_diff:.1f}K (threshold: 8.6K), rpm={rpm:.0f} (threshold: 1380)"
            })

        # Power Failure (PWF)
        # Risk when: power < 3500 OR power > 9000
        power = row["power"]
        if power < 3500:
            severity = min(1.0, (3500 - power) / 3500)
            failures.append({
                "type": "Power Failure (PWF) - Low Power",
                "severity": severity,
                "severity_label": get_severity_label(severity),
                "details": f"power={power:.0f}W (threshold: >3500W)"
            })
        elif power > 9000:
            severity = min(1.0, (power - 9000) / 3000)
            failures.append({
                "type": "Power Failure (PWF) - High Power",
                "severity": severity,
                "severity_label": get_severity_label(severity),
                "details": f"power={power:.0f}W (threshold: <9000W)"
            })

        # Overstrain Failure (OSF)
        # Risk when: strain > 11000
        strain = row["strain"]
        if strain > 11000:
            severity = min(1.0, (strain - 11000) / 5000)
            failures.append({
                "type": "Overstrain Failure (OSF)",
                "severity": severity,
                "severity_label": get_severity_label(severity),
                "details": f"strain={strain:.0f} (threshold: 11000)"
            })

        # Tool Wear Failure (TWF)
        # Risk when: tool_wear >= 200 (critical at 240)
        tool_wear = row["Tool wear [min]"]
        if tool_wear >= 200:
            severity = min(1.0, (tool_wear - 200) / 40)
            failures.append({
                "type": "Tool Wear Failure (TWF)",
                "severity": severity,
                "severity_label": get_severity_label(severity),
                "details": f"tool_wear={tool_wear:.0f}min (threshold: 200min, critical: 240min)"
            })

        # Sort by severity (highest first)
        failures.sort(key=lambda x: x["severity"], reverse=True)

        results.append({
            "failure_types": failures,
            "primary_failure": failures[0] if failures else None,
            "failure_count": len(failures)
        })

    return results

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

# Test health score and failure type detection
print("\n" + "="*50)
print("HEALTH ASSESSMENT EXAMPLES")
print("="*50)

# Sample some test cases (mix of healthy and unhealthy)
# Get indices of healthy and unhealthy samples
healthy_indices = X_test[y_test == 0].index[:2].tolist()
unhealthy_indices = X_test[y_test == 1].index[:3].tolist()
sample_indices = healthy_indices + unhealthy_indices

X_sample = X_test.loc[sample_indices]
y_sample = y_test.loc[sample_indices]

scores, probas = compute_health_score(model, X_sample)
failure_results = detect_failure_types(X_sample)

print("\nDetailed Health Assessment:")
print("-" * 70)
for i, (idx, score, proba, actual, failures) in enumerate(zip(
    sample_indices, scores, probas, y_sample, failure_results
)):
    status = "Unhealthy" if actual == 1 else "Healthy"
    print(f"\nSample {i+1} (Actual: {status})")
    print(f"  Health Score: {score:.1f}/10")
    print(f"  Unhealthy Probability: {proba*100:.1f}%")

    if failures["failure_count"] > 0:
        print(f"  Detected Issues ({failures['failure_count']}):")
        for f in failures["failure_types"]:
            print(f"    - {f['type']}")
            print(f"      Severity: {f['severity_label']} ({f['severity']*100:.0f}%)")
            print(f"      Details: {f['details']}")
    else:
        print("  Detected Issues: None - Machine is healthy")

# -------------------------------------------------
# 12. Save model and metadata
# -------------------------------------------------
os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)

# Save model
joblib.dump(model, os.path.join(BASE_DIR, "models", "health_classification_model.pkl"))

# Save feature columns for inference
joblib.dump(feature_cols, os.path.join(BASE_DIR, "models", "health_feature_cols.pkl"))

# Save health score and failure detection configuration
health_config = {
    "risk_cols": ["hdf_risk_bin", "pwf_risk_bin", "osf_risk_bin", "twf_risk_bin"],
    "score_weights": {"probability": 0.5, "risk_severity": 0.5},
    "score_range": (1, 10),
    "failure_thresholds": {
        "HDF": {"temp_diff": 8.6, "rpm": 1380},
        "PWF": {"power_min": 3500, "power_max": 9000},
        "OSF": {"strain": 11000},
        "TWF": {"tool_wear_warning": 200, "tool_wear_critical": 240}
    },
    "severity_labels": {
        "critical": 0.7,
        "high": 0.4,
        "moderate": 0.2,
        "low": 0.0
    }
}
joblib.dump(health_config, os.path.join(BASE_DIR, "models", "health_config.pkl"))

print("\n" + "="*50)
print("MODEL SAVED")
print("="*50)
print("\n✅ Health classification model trained and saved")
print("   - health_classification_model.pkl (Gradient Boosting classifier)")
print("   - health_feature_cols.pkl (16 feature columns)")
print("   - health_config.pkl (thresholds and configuration)")
print("\nCapabilities:")
print("   1. Binary health classification (Healthy/Unhealthy)")
print("   2. Health score (1-10 scale)")
print("   3. Failure type detection with severity levels")

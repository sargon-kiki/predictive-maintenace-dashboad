import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
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
# 3. Create Failure Type labels (multi-class)
# -------------------------------------------------
def get_failure_type(row):
    """
    Determine failure type from one-hot encoded columns.
    Priority order handles cases where multiple failures occur.
    """
    if row["TWF"] == 1:
        return "Tool Wear Failure"
    if row["HDF"] == 1:
        return "Heat Dissipation Failure"
    if row["PWF"] == 1:
        return "Power Failure"
    if row["OSF"] == 1:
        return "Overstrain Failure"
    if row["RNF"] == 1:
        return "Random Failure"
    return "No Failure"

df["Failure_Type"] = df.apply(get_failure_type, axis=1)

# Print class distribution
print("\nFailure Type Distribution:")
print(df["Failure_Type"].value_counts())

# -------------------------------------------------
# 4. Physics-based feature engineering
# -------------------------------------------------
# Temperature difference
df["temp_diff"] = df["Process temperature [K]"] - df["Air temperature [K]"]

# Power = Torque × Angular velocity
df["power"] = df["Torque [Nm]"] * df["Rotational speed [rpm]"] * 2 * np.pi / 60

# Strain = Tool wear × Torque
df["strain"] = df["Tool wear [min]"] * df["Torque [Nm]"]

# Risk indicators (binary)
df["hdf_risk"] = ((df["temp_diff"] < 8.6) & (df["Rotational speed [rpm]"] < 1380)).astype(int)
df["pwf_risk"] = ((df["power"] < 3500) | (df["power"] > 9000)).astype(int)
df["osf_risk"] = (df["strain"] > 11000).astype(int)
df["twf_risk"] = (df["Tool wear [min]"] >= 200).astype(int)
df["risk_score"] = df["hdf_risk"] + df["pwf_risk"] + df["osf_risk"] + df["twf_risk"]

# One-hot encode product Type
df = pd.get_dummies(df, columns=["Type"], prefix="Type")

print("\nPhysics-based features engineered")

# -------------------------------------------------
# 5. Define features and target
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
    "hdf_risk",
    "pwf_risk",
    "osf_risk",
    "twf_risk",
    "risk_score",
    "Type_H",
    "Type_L",
    "Type_M"
]

X = df[feature_cols]
y = df["Failure_Type"]

# Get class names for later use
class_names = [
    "No Failure",
    "Tool Wear Failure",
    "Heat Dissipation Failure",
    "Power Failure",
    "Overstrain Failure",
    "Random Failure"
]

# -------------------------------------------------
# 6. Train-test split
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
# 7. Handle class imbalance with SMOTE
# -------------------------------------------------
print("\nApplying SMOTE to handle class imbalance...")
smote = SMOTE(random_state=42, k_neighbors=3)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(f"Original training set: {len(X_train)} samples")
print(f"Resampled training set: {len(X_train_resampled)} samples")
print("\nResampled class distribution:")
print(pd.Series(y_train_resampled).value_counts())

# -------------------------------------------------
# 8. Train Random Forest Classifier (multi-class)
# -------------------------------------------------
model = RandomForestClassifier(
    n_estimators=500,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

print("\nTraining multi-class failure prediction model...")
model.fit(X_train_resampled, y_train_resampled)

# -------------------------------------------------
# 9. Evaluate model
# -------------------------------------------------
y_pred = model.predict(X_test)

print("\n" + "="*60)
print("MODEL EVALUATION")
print("="*60)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nOverall Accuracy: {accuracy*100:.2f}%")

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred, labels=class_names)
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, labels=class_names, zero_division=0))

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
print(f"Cross-Validation Accuracy: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")

# Feature importance
print("\nTop 10 Feature Importances:")
importance_df = pd.DataFrame({
    "feature": feature_cols,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

for _, row in importance_df.head(10).iterrows():
    bar = "█" * int(row["importance"] * 50)
    print(f"  {row['feature']:25s} {row['importance']:.4f} {bar}")

# -------------------------------------------------
# 10. Per-class accuracy analysis
# -------------------------------------------------
print("\n" + "="*60)
print("PER-CLASS PERFORMANCE")
print("="*60)

for class_name in class_names:
    mask = y_test == class_name
    if mask.sum() > 0:
        class_acc = (y_pred[mask] == y_test[mask]).mean()
        count = mask.sum()
        print(f"  {class_name:25s}: {class_acc*100:6.2f}% ({count} samples)")

# -------------------------------------------------
# 11. Sample predictions
# -------------------------------------------------
print("\n" + "="*60)
print("SAMPLE PREDICTIONS")
print("="*60)

# Get samples from each class
print("\n{:^8} {:^25} {:^25} {:^10}".format("Sample", "Actual", "Predicted", "Correct"))
print("-" * 75)

shown_classes = set()
for idx in range(len(X_test)):
    actual = y_test.iloc[idx]
    if actual not in shown_classes:
        predicted = y_pred[idx]
        correct = "✓" if actual == predicted else "✗"
        print(f"{idx:^8} {actual:^25} {predicted:^25} {correct:^10}")
        shown_classes.add(actual)
    if len(shown_classes) == 6:
        break

# -------------------------------------------------
# 12. Save model and metadata
# -------------------------------------------------
os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)

# Save model
joblib.dump(model, os.path.join(BASE_DIR, "models", "failure_prediction_model.pkl"))

# Save feature columns
joblib.dump(feature_cols, os.path.join(BASE_DIR, "models", "failure_feature_cols.pkl"))

# Save class names and configuration
failure_config = {
    "class_names": class_names,
    "feature_cols": feature_cols,
    "failure_descriptions": {
        "No Failure": "Machine is operating normally",
        "Tool Wear Failure": "Tool has exceeded wear threshold (200-240 min)",
        "Heat Dissipation Failure": "Insufficient heat dissipation (temp_diff < 8.6K, RPM < 1380)",
        "Power Failure": "Power outside normal range (< 3500W or > 9000W)",
        "Overstrain Failure": "Mechanical overstrain (strain > 11000)",
        "Random Failure": "Random/unexplained failure event"
    }
}
joblib.dump(failure_config, os.path.join(BASE_DIR, "models", "failure_config.pkl"))

print("\n" + "="*60)
print("MODEL SAVED")
print("="*60)
print("\n✅ Multi-class failure prediction model trained and saved")
print("   - failure_prediction_model.pkl (Gradient Boosting Classifier)")
print("   - failure_feature_cols.pkl (16 feature columns)")
print("   - failure_config.pkl (class names and descriptions)")
print("\nCapabilities:")
print("   - Predicts 6 classes: No Failure + 5 failure types")
print("   - Provides confidence probabilities for each class")
print("   - Uses physics-based features for accurate prediction")

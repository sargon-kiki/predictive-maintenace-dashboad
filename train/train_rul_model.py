import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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
# Temperature difference (thermal stress indicator)
df["temp_diff"] = df["Process temperature [K]"] - df["Air temperature [K]"]

# Power = Torque × Angular velocity
df["power"] = df["Torque [Nm]"] * df["Rotational speed [rpm]"] * 2 * np.pi / 60

# Strain = Tool wear × Torque (mechanical stress)
df["strain"] = df["Tool wear [min]"] * df["Torque [Nm]"]

# Normalized features for degradation calculation
df["temp_diff_norm"] = df["temp_diff"] / df["temp_diff"].max()
df["power_norm"] = df["power"] / df["power"].max()
df["torque_norm"] = df["Torque [Nm]"] / df["Torque [Nm]"].max()
df["rpm_norm"] = df["Rotational speed [rpm]"] / df["Rotational speed [rpm]"].max()

print("Physics-based features engineered")

# -------------------------------------------------
# 4. Risk indicators (stress factors)
# -------------------------------------------------
# Heat Dissipation Failure risk
df["hdf_risk"] = ((df["temp_diff"] < 8.6) & (df["Rotational speed [rpm]"] < 1380)).astype(int)

# Power Failure risk (operating outside optimal range)
df["pwf_risk"] = ((df["power"] < 3500) | (df["power"] > 9000)).astype(int)

# Overstrain Failure risk
df["osf_risk"] = (df["strain"] > 11000).astype(int)

# Tool Wear Failure risk (approaching critical wear)
df["twf_risk"] = (df["Tool wear [min]"] >= 200).astype(int)

# Combined risk count
df["risk_count"] = df["hdf_risk"] + df["pwf_risk"] + df["osf_risk"] + df["twf_risk"]

# -------------------------------------------------
# 5. Compute Degradation-Adjusted RUL
# -------------------------------------------------
MAX_TOOL_WEAR = df["Tool wear [min]"].max()  # 253 minutes
print(f"\nMax tool wear in dataset: {MAX_TOOL_WEAR} minutes")

# Base RUL (simple remaining tool life)
df["base_rul"] = MAX_TOOL_WEAR - df["Tool wear [min]"]

# Degradation rate based on operating stress
# Higher stress = faster degradation = multiplier > 1
# Formula: degradation_rate = 1 + stress_factors

def compute_degradation_rate(row):
    """
    Compute degradation rate multiplier based on operating conditions.
    Returns value >= 1.0, where higher values mean faster wear.
    """
    rate = 1.0

    # Thermal stress: high temp diff increases wear
    # Optimal temp_diff is around 10K, deviation increases degradation
    temp_stress = abs(row["temp_diff"] - 10) / 10
    rate += temp_stress * 0.3  # Up to 30% increase

    # Power stress: operating outside optimal range (4000-8000W)
    power = row["power"]
    if power < 4000:
        power_stress = (4000 - power) / 4000
        rate += power_stress * 0.4  # Low power = potential issues
    elif power > 8000:
        power_stress = (power - 8000) / 4000
        rate += power_stress * 0.4  # High power = faster wear

    # Torque stress: high torque accelerates wear
    if row["Torque [Nm]"] > 50:
        torque_stress = (row["Torque [Nm]"] - 50) / 30
        rate += torque_stress * 0.3

    # RPM stress: very high or very low RPM
    rpm = row["Rotational speed [rpm]"]
    if rpm < 1300:
        rate += 0.2  # Low RPM can indicate issues
    elif rpm > 2000:
        rpm_stress = (rpm - 2000) / 1000
        rate += rpm_stress * 0.2  # High RPM = faster wear

    # Risk factor penalty: each active risk adds to degradation
    rate += row["risk_count"] * 0.25

    return max(rate, 1.0)  # Minimum rate is 1.0

df["degradation_rate"] = df.apply(compute_degradation_rate, axis=1)

# Adjusted RUL = Base RUL / Degradation Rate
# Higher degradation rate = shorter actual RUL
df["RUL_hours"] = df["base_rul"] / df["degradation_rate"]

# Ensure non-negative RUL
df["RUL_hours"] = df["RUL_hours"].clip(lower=0)

print(f"RUL range: {df['RUL_hours'].min():.1f} - {df['RUL_hours'].max():.1f} hours")
print(f"Mean RUL: {df['RUL_hours'].mean():.1f} hours")
print(f"Degradation rate range: {df['degradation_rate'].min():.2f} - {df['degradation_rate'].max():.2f}")

# -------------------------------------------------
# 6. One-hot encode product Type
# -------------------------------------------------
df = pd.get_dummies(df, columns=["Type"], prefix="Type")

# -------------------------------------------------
# 7. Define features and target
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

X = df[feature_cols]
y = df["RUL_hours"]

# -------------------------------------------------
# 8. Train-test split
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print(f"\nTraining set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# -------------------------------------------------
# 9. Train Gradient Boosting Regressor
# -------------------------------------------------
model = GradientBoostingRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

model.fit(X_train, y_train)

# -------------------------------------------------
# 10. Evaluate model
# -------------------------------------------------
y_pred = model.predict(X_test)

print("\n" + "="*50)
print("MODEL EVALUATION")
print("="*50)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100  # Avoid div by zero

print(f"\nMAE (Mean Absolute Error): {mae:.2f} hours")
print(f"RMSE (Root Mean Square Error): {rmse:.2f} hours")
print(f"R² Score: {r2:.4f}")
print(f"MAPE (Mean Absolute Percentage Error): {mape:.2f}%")

# Cross-validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=cv, scoring="r2")
print(f"\nCross-Validation R²: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

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
# 11. Sample Predictions
# -------------------------------------------------
print("\n" + "="*50)
print("SAMPLE PREDICTIONS")
print("="*50)

# Select samples across the RUL range
sample_indices = [
    y_test.idxmin(),  # Lowest RUL
    y_test.idxmax(),  # Highest RUL
]
# Add some middle samples
sorted_test = y_test.sort_values()
sample_indices.extend([
    sorted_test.index[len(sorted_test)//4],
    sorted_test.index[len(sorted_test)//2],
    sorted_test.index[3*len(sorted_test)//4],
])

print("\n{:^10} {:^15} {:^15} {:^10}".format("Sample", "Actual RUL", "Predicted RUL", "Error"))
print("-" * 55)

for idx in sample_indices[:5]:
    actual = y_test.loc[idx]
    predicted = model.predict(X_test.loc[[idx]])[0]
    error = abs(actual - predicted)
    print(f"{str(idx):^10} {actual:^15.1f} {predicted:^15.1f} {error:^10.1f}")

# -------------------------------------------------
# 12. RUL Category Analysis
# -------------------------------------------------
print("\n" + "="*50)
print("RUL CATEGORY ACCURACY")
print("="*50)

def categorize_rul(rul):
    if rul < 20:
        return "Critical (<20h)"
    elif rul < 50:
        return "Warning (20-50h)"
    elif rul < 100:
        return "Moderate (50-100h)"
    else:
        return "Good (>100h)"

y_test_cat = y_test.apply(categorize_rul)
y_pred_cat = pd.Series(y_pred, index=y_test.index).apply(categorize_rul)

category_accuracy = (y_test_cat == y_pred_cat).mean()
print(f"\nCategory Classification Accuracy: {category_accuracy*100:.2f}%")

# Per-category breakdown
print("\nPer-Category Performance:")
for cat in ["Critical (<20h)", "Warning (20-50h)", "Moderate (50-100h)", "Good (>100h)"]:
    mask = y_test_cat == cat
    if mask.sum() > 0:
        cat_mae = mean_absolute_error(y_test[mask], y_pred[mask.values])
        cat_acc = (y_test_cat[mask] == y_pred_cat[mask]).mean()
        print(f"  {cat:20s}: MAE={cat_mae:.1f}h, Category Acc={cat_acc*100:.1f}%")

# -------------------------------------------------
# 13. Save model and metadata
# -------------------------------------------------
os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)

# Save model
joblib.dump(model, os.path.join(BASE_DIR, "models", "rul_model.pkl"))

# Save feature columns
joblib.dump(feature_cols, os.path.join(BASE_DIR, "models", "rul_feature_cols.pkl"))

# Save configuration
rul_config = {
    "max_tool_wear": MAX_TOOL_WEAR,
    "feature_cols": feature_cols,
    "rul_categories": {
        "critical": 20,
        "warning": 50,
        "moderate": 100
    }
}
joblib.dump(rul_config, os.path.join(BASE_DIR, "models", "rul_config.pkl"))

print("\n" + "="*50)
print("MODEL SAVED")
print("="*50)
print("\n✅ RUL model trained and saved")
print("   - rul_model.pkl (Gradient Boosting Regressor)")
print("   - rul_feature_cols.pkl (21 feature columns)")
print("   - rul_config.pkl (configuration)")
print("\nCapabilities:")
print("   1. Degradation-adjusted RUL prediction")
print("   2. Accounts for operating stress factors")
print("   3. Risk-aware remaining life estimation")

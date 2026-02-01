import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import numpy as np

# -------------------------------------------------
# 1. Load dataset safely
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "ai4i_2020.csv")

df = pd.read_csv(DATA_PATH)
print("Dataset loaded successfully")

# -------------------------------------------------
# 2. Drop non-informative columns
# -------------------------------------------------
df = df.drop(columns=["UDI", "Product ID"])

# -------------------------------------------------
# 3. Define RUL target in HOURS
# -------------------------------------------------
MAX_TOOL_WEAR = df["Tool wear [min]"].max()

df["RUL_hours"] = (MAX_TOOL_WEAR - df["Tool wear [min]"]) 

# -------------------------------------------------
# 4. Define features and target
# -------------------------------------------------
X = df[[
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]]

y = df["RUL_hours"]

# -------------------------------------------------
# 5. Train-test split
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -------------------------------------------------
# 6. Train Random Forest Regressor
# -------------------------------------------------
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# -------------------------------------------------
# 7. Evaluate model
# -------------------------------------------------
y_pred = model.predict(X_test)

print("\nRUL Model Evaluation:")
print("MAE (hours):", mean_absolute_error(y_test, y_pred))
print("RMSE (hours):", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R² Score:", r2_score(y_test, y_pred))

# -------------------------------------------------
# 8. Save trained model
# -------------------------------------------------
os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
joblib.dump(model, os.path.join(BASE_DIR, "models", "rul_model.pkl"))

print("\n✅ RUL model trained and saved (hours)")

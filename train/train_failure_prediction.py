import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

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
# 3. Define features and target
# -------------------------------------------------
X = df[[
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]]

y = df["Machine failure"]

# -------------------------------------------------
# 4. Train-test split
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------------------------------------
# 5. Train Random Forest classifier
# -------------------------------------------------
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# -------------------------------------------------
# 6. Evaluate model
# -------------------------------------------------
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# -------------------------------------------------
# 7. Save trained model
# -------------------------------------------------
os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
joblib.dump(model, os.path.join(BASE_DIR, "models", "failure_prediction_model.pkl"))

print("\n✅ Binary failure prediction model trained and saved")

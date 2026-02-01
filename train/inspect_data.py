import os
import pandas as pd

# Build safe path to data file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "ai4i_2020.csv")

# Load dataset
df = pd.read_csv(DATA_PATH)

# Basic checks
print("Dataset loaded successfully")
print("\nShape (rows, columns):")
print(df.shape)

print("\nColumn names:")
for col in df.columns:
    print(col)

print("\nFirst 5 rows:")
print(df.head())

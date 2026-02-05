import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample, shuffle
import joblib

# -----------------------
# Load and Merge CICIDS2017 Dataset
# -----------------------
dataset_dir = "C:\\Users\\jahma\\OneDrive\\Desktop\\Datasets\\MachineLearningCSV"
csv_files = [f for f in os.listdir(dataset_dir) if f.endswith(".csv")]
df_list = [pd.read_csv(os.path.join(dataset_dir, file)) for file in csv_files]
df_cic = pd.concat(df_list, ignore_index=True)

print(f"✅ Loaded {len(csv_files)} CSV files. Total records: {df_cic.shape[0]}")

# Drop Unnecessary Columns
drop_cols = ["Flow ID", "Source IP", "Destination IP", "Timestamp"]
df_cic.drop(columns=[col for col in drop_cols if col in df_cic.columns], inplace=True)

df_cic.columns = df_cic.columns.str.strip()
if "Label" not in df_cic.columns:
    raise ValueError("❌ Label column not found in CICIDS2017 dataset!")

# Encode Labels for Multi-Class Classification
label_encoder = LabelEncoder()
df_cic["Label"] = label_encoder.fit_transform(df_cic["Label"])
print("✅ CICIDS2017 Labels Encoded")

# Handle Missing/Infinite Values
df_cic.replace([np.inf, -np.inf], np.nan, inplace=True)
df_cic.dropna(inplace=True)

# 1. Fit on raw data
scaler = StandardScaler()
scaler.fit(df_cic.iloc[:, :-1])

# 2. Save the fitted scaler
joblib.dump(scaler, "fitted_scaler.pkl")

# 3. Transform the raw data for training
df_cic.iloc[:, :-1] = scaler.transform(df_cic.iloc[:, :-1]).astype(np.float64)

# Train-Test Split
X_train_cic, X_test_cic, y_train_cic, y_test_cic = train_test_split(
    df_cic.drop("Label", axis=1), df_cic["Label"], test_size=0.2, random_state=42
)

# Save Processed CICIDS2017 Data
output_dir = os.path.join(dataset_dir, "Processed")
os.makedirs(output_dir, exist_ok=True)
X_train_cic.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
X_test_cic.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
y_train_cic.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
y_test_cic.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)
print("✅ CICIDS2017 Preprocessing Complete!")

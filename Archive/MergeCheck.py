import pandas as pd
import os

dataset_dir = "C:\\Users\\jahma\\OneDrive\\Desktop\\Datasets\\Processed"

y_train_cic = pd.read_csv(os.path.join(dataset_dir, "y_train_cic.csv"))
X_train_cic = pd.read_csv(os.path.join(dataset_dir, "X_train_cic.csv"))
X_train_unsw = pd.read_csv(os.path.join(dataset_dir, "X_train_unsw_aligned.csv"))
X_train_unsw = pd.read_csv(os.path.join(dataset_dir, "X_train_unsw_aligned.csv"))
y_train_unsw = pd.read_csv(os.path.join(dataset_dir, "y_train_unsw_encoded.csv"))

print("🔍 Missing values in X_train_unsw:", X_train_unsw.isnull().sum().sum())
print("🔍 Missing values in y_train_unsw:", y_train_unsw.isnull().sum().sum())

print("Final CICIDS Train Features Shape:", X_train_cic.shape)
print("Final UNSW Train Features Shape:", X_train_unsw.shape)

# Ensure both datasets have the same columns
if set(X_train_cic.columns) == set(X_train_unsw.columns):
    print("✅ Features Match Exactly!")
else:
    missing_in_unsw = set(X_train_cic.columns) - set(X_train_unsw.columns)
    extra_in_unsw = set(X_train_unsw.columns) - set(X_train_cic.columns)
    print(f"⚠️ Missing in UNSW: {missing_in_unsw}")
    print(f"⚠️ Extra in UNSW: {extra_in_unsw}")

print("🎯 Unique classes in y_train_unsw:", sorted(y_train_unsw["Label"].unique()))
print("🎯 Unique classes in y_train_cic:", sorted(y_train_cic["Label"].unique()))
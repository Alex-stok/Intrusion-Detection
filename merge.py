import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

# -------------------------------
# Load CICIDS Data
# -------------------------------
cic_x_train = pd.read_csv("C:\\Users\\jahma\\OneDrive\\Desktop\\Datasets\\Processed\\X_train.csv")
cic_x_test = pd.read_csv("C:\\Users\\jahma\\OneDrive\\Desktop\\Datasets\\Processed\\X_test.csv")
cic_y_train = pd.read_csv("C:\\Users\\jahma\\OneDrive\\Desktop\\Datasets\\Processed\\y_train.csv")
cic_y_test = pd.read_csv("C:\\Users\\jahma\\OneDrive\\Desktop\\Datasets\\Processed\\y_test.csv")

# -------------------------------
# Load UNSW-NB15 Data
# -------------------------------
unsw_x_train = pd.read_csv("C:\\Users\\jahma\\OneDrive\\Desktop\\Datasets\\Processed\\Set 2\\X_train_unsw.csv")
unsw_x_test = pd.read_csv("C:\\Users\\jahma\\OneDrive\\Desktop\\Datasets\\Processed\\Set 2\\X_test_unsw.csv")
unsw_y_train = pd.read_csv("C:\\Users\\jahma\\OneDrive\\Desktop\\Datasets\\Processed\\Set 2\\y_train_unsw.csv")
unsw_y_test = pd.read_csv("C:\\Users\\jahma\\OneDrive\\Desktop\\Datasets\\Processed\\Set 2\\y_test_unsw.csv")

# -------------------------------
# 1. Align Features (Fill Missing Features with 0)
# -------------------------------
all_features = set(cic_x_train.columns).union(set(unsw_x_train.columns))

for feature in all_features:
    if feature not in cic_x_train.columns:
        cic_x_train[feature] = 0
        cic_x_test[feature] = 0
    if feature not in unsw_x_train.columns:
        unsw_x_train[feature] = 0
        unsw_x_test[feature] = 0

# Reorder columns for consistency
cic_x_train = cic_x_train[sorted(all_features)]
cic_x_test = cic_x_test[sorted(all_features)]
unsw_x_train = unsw_x_train[sorted(all_features)]
unsw_x_test = unsw_x_test[sorted(all_features)]

# -------------------------------
# 2. Unify Attack Labels
# -------------------------------
cic_label_map = {
    0: "Benign", 1: "Brute Force", 2: "DoS/DDoS", 3: "DoS/DDoS", 4: "DoS/DDoS",
    5: "DoS/DDoS", 6: "DoS/DDoS", 7: "Brute Force", 8: "Exploits", 9: "Exploits",
    10: "Reconnaissance", 11: "Brute Force", 12: "Exploits", 13: "Exploits", 14: "Exploits"
}

unsw_label_map = {
    "Normal": "Benign", "Generic": "Brute Force", "Exploits": "Exploits", 
    "Fuzzers": "Fuzzing", "DoS": "DoS/DDoS", "Reconnaissance": "Reconnaissance",
    "Analysis": "Analysis", "Backdoor": "Backdoor", "Shellcode": "Exploits", "Worms": "Worms"
}

# Apply mappings
cic_y_train["Label"] = cic_y_train["Label"].map(cic_label_map)
cic_y_test["Label"] = cic_y_test["Label"].map(cic_label_map)
unsw_y_train["Label"] = unsw_y_train["Label"].map(unsw_label_map)
unsw_y_test["Label"] = unsw_y_test["Label"].map(unsw_label_map)

# Ensure no NaNs exist in labels
cic_y_train.dropna(inplace=True)
cic_y_test.dropna(inplace=True)
unsw_y_train.dropna(inplace=True)
unsw_y_test.dropna(inplace=True)

# -------------------------------
# 3. Merge the Datasets Separately
# -------------------------------
X_train_merged = pd.concat([cic_x_train, unsw_x_train], ignore_index=True)
X_test_merged = pd.concat([cic_x_test, unsw_x_test], ignore_index=True)
y_train_merged = pd.concat([cic_y_train, unsw_y_train], ignore_index=True)
y_test_merged = pd.concat([cic_y_test, unsw_y_test], ignore_index=True)

# -------------------------------
# 4. Ensure Alignment of X and Y
# -------------------------------
# Ensure label count matches feature count
if len(y_train_merged) != len(X_train_merged):
    print(f"⚠️ Mismatch detected! Adjusting y_train_merged from {len(y_train_merged)} to {len(X_train_merged)}")
    y_train_merged = y_train_merged.iloc[:len(X_train_merged)]

if len(y_test_merged) != len(X_test_merged):
    print(f"⚠️ Mismatch detected! Adjusting y_test_merged from {len(y_test_merged)} to {len(X_test_merged)}")
    y_test_merged = y_test_merged.iloc[:len(X_test_merged)]

# Re-check alignment before saving
assert len(X_train_merged) == len(y_train_merged), "❌ Mismatch still exists in training data!"
assert len(X_test_merged) == len(y_test_merged), "❌ Mismatch still exists in test data!"

# -------------------------------
# 5. Save the Merged Datasets
# -------------------------------
output_dir = "C:\\Users\\jahma\\OneDrive\\Desktop\\Datasets\\Merged"
os.makedirs(output_dir, exist_ok=True)

X_train_merged.to_csv(os.path.join(output_dir, "X_train_merged.csv"), index=False)
X_test_merged.to_csv(os.path.join(output_dir, "X_test_merged.csv"), index=False)
y_train_merged.to_csv(os.path.join(output_dir, "y_train_merged.csv"), index=False)
y_test_merged.to_csv(os.path.join(output_dir, "y_test_merged.csv"), index=False)

print("✅ Train and Test sets merged and saved separately, with missing values handled!")

import os
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------
# 🔧 Configuration
# -----------------------
dataset_dir = "C:\\Users\\jahma\\Downloads\\CSV-01-12\\01-12"
benign_file = "C:\\Users\\jahma\\OneDrive\\Desktop\\Datasets\\MachineLearningCSV\\Monday-WorkingHours.pcap_ISCX.csv"
model_output_path = os.path.join(dataset_dir, "new_cicddos2019.model")

selected_files = ["DrDoS_UDP.csv", "Syn.csv", "DrDoS_NTP.csv"]
csv_files = [os.path.join(dataset_dir, f) for f in selected_files if os.path.exists(os.path.join(dataset_dir, f))]

print("🚀 Loading and sampling 500,000 rows from each dataset...")

merged_data = []
for file in csv_files:
    label_name = os.path.splitext(os.path.basename(file))[0]
    print(f"📄 Reading {os.path.basename(file)} with label '{label_name}'...")
    df = pd.read_csv(file, low_memory=False)
    df.columns = df.columns.str.strip()
    df = df.sample(n=500000, random_state=42)
    df['Label'] = label_name
    df_numeric = df.select_dtypes(include=['int64', 'float64'])
    df = df[df_numeric.map(np.isfinite).all(axis=1)]
    merged_data.append(df)

print("📄 Reading and sampling benign samples...")
benign_df = pd.read_csv(benign_file, low_memory=False)
benign_df.columns = benign_df.columns.str.strip()
benign_df = benign_df[benign_df['Label'] == 'BENIGN']
benign_df = benign_df.sample(n=500000, random_state=42)
benign_numeric = benign_df.select_dtypes(include=['int64', 'float64'])
benign_df = benign_df[benign_numeric.map(np.isfinite).all(axis=1)]
benign_df['Label'] = 'BENIGN'
print(f"📄 BENIGN valid rows: {len(benign_df)}")


# Drop non-numeric or identifier columns in all attack datasets
print("✅ Dropping columns:")
drop_cols = ['Flow ID', 'Source IP', 'Destination IP', 'Timestamp', 'Unnamed: 0', 'SimillarHTTP', 'Inbound', 'Protocol', 'Source Port', 'Destinatiion IP']
for i in range(len(merged_data)):
    merged_data[i] = merged_data[i].drop(columns=[col for col in drop_cols if col in merged_data[i].columns], errors='ignore')

# Drop same from benign
benign_df = benign_df.drop(columns=[col for col in drop_cols if col in benign_df.columns], errors='ignore')


print("✅ Merging:")
# ✅ Align all datasets to share the same columns
base_columns = merged_data[0].columns  # Use first attack file's columns as schema
for i in range(len(merged_data)):
    merged_data[i] = merged_data[i].reindex(columns=base_columns)

benign_df = benign_df.reindex(columns=base_columns)
merged_data.append(benign_df)

# ✅ Combine all dataframes
full_df = pd.concat(merged_data, ignore_index=True)
print("✅ Final label distribution after merging:")
print(full_df['Label'].value_counts())

# ✅ Filter rows with only finite numeric values
X_full = full_df.select_dtypes(include=['int64', 'float64'])
finite_mask = np.isfinite(X_full).all(axis=1)

X = X_full[finite_mask]
y = full_df.loc[finite_mask, 'Label']

print("✅ Label counts before encoding:")
print(y.value_counts())

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print("✅ Encoded label classes:", list(label_encoder.classes_))
print("📦 Encoded label distribution:", pd.Series(y_encoded).value_counts())

# Drop constant features
constant_cols = X.columns[X.nunique() <= 1]
if len(constant_cols) > 0:
    print(f"🚨 Dropping constant features: {list(constant_cols)}")
    X.drop(columns=constant_cols, inplace=True)

# Feature selection
selector = SelectKBest(score_func=f_classif, k=50)
X_selected = selector.fit_transform(X, y_encoded)
selected_features = X.columns[selector.get_support()]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_selected, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# GridSearchCV
param_grid = {
    'learning_rate': [0.1],
    'max_depth': [6],
    'subsample': [0.8],
    'colsample_bytree': [0.8],
    'n_estimators': [100]
}

gscv = GridSearchCV(
    estimator=xgb.XGBClassifier(objective='multi:softprob', num_class=len(label_encoder.classes_),
                                tree_method='hist', eval_metric='mlogloss', use_label_encoder=False),
    param_grid=param_grid, scoring='f1_weighted', cv=3, verbose=1, n_jobs=1
)

gscv.fit(X_train, y_train)
print(f"✅ Best parameters: {gscv.best_params_}")
model = gscv.best_estimator_

# Save model
model.save_model(model_output_path)
print(f"✅ Model saved to {model_output_path}")

# Evaluation
y_pred = model.predict(X_test)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

print("📦 y_test class counts:")
print(pd.Series(y_test).value_counts())
print("✅ Label mapping:", dict(enumerate(label_encoder.classes_)))

print("📉 Confusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# Feature importance
xgb.plot_importance(model, max_num_features=20, importance_type='gain')
plt.title("Top 20 Feature Importances")
plt.tight_layout()
plt.show()

joblib.dump((selector, X.columns.tolist()), "selector.pkl")
print("✅ Selector saved.")

# ---------------------------------------------
# 📊 Accuracy and Log Loss Line Graphs
# ---------------------------------------------

# Split unselected data for visual evaluation tracking
X_train_full, X_valid_full, y_train_full, y_valid_full = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

dtrain = xgb.DMatrix(X_train_full, label=y_train_full)
dvalid = xgb.DMatrix(X_valid_full, label=y_valid_full)

params = {
    'objective': 'multi:softprob',
    'num_class': len(label_encoder.classes_),
    'eval_metric': ['mlogloss', 'merror'],
    'learning_rate': 0.1,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8
}

evals_result = {}
booster = xgb.train(
    params,
    dtrain,
    num_boost_round=100,
    evals=[(dtrain, "Train"), (dvalid, "Validation")],
    evals_result=evals_result,
    verbose_eval=False
)

# Plot Log Loss
plt.figure(figsize=(10, 5))
plt.plot(evals_result['Train']['mlogloss'], label='Train Log Loss')
plt.plot(evals_result['Validation']['mlogloss'], label='Validation Log Loss')
plt.xlabel('Boosting Round')
plt.ylabel('Log Loss')
plt.title('Log Loss over Boosting Rounds')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot Classification Error (1 - Accuracy)
plt.figure(figsize=(10, 5))
plt.plot(evals_result['Train']['merror'], label='Train Error')
plt.plot(evals_result['Validation']['merror'], label='Validation Error')
plt.xlabel('Boosting Round')
plt.ylabel('Classification Error')
plt.title('Error Rate over Boosting Rounds (1 - Accuracy)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
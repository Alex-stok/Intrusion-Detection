import os
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

# -------------------------------
# 📌 Define Paths
# -------------------------------
dataset_dir = "C:\\Users\\jahma\\OneDrive\\Desktop\\Datasets\\Processed"
model_cicids_path = os.path.join(dataset_dir, "xgboost_cicids.model")
model_finetuned_path = os.path.join(dataset_dir, "xgboost_combined.model")

# -------------------------------
# 📌 Load Test Datasets
# -------------------------------
print("🚀 Loading test datasets...")
X_test_cic = pd.read_csv(os.path.join(dataset_dir, "X_test_cic.csv"))
y_test_cic = pd.read_csv(os.path.join(dataset_dir, "y_test_cic.csv"))

X_test_unsw = pd.read_csv(os.path.join(dataset_dir, "X_test_unsw_aligned.csv"))
y_test_unsw = pd.read_csv(os.path.join(dataset_dir, "y_test_unsw_encoded.csv"))

# Ensure labels are integers
y_test_cic = y_test_cic["Label"].astype(int).values
y_test_unsw = y_test_unsw["Label"].astype(int).values

# Determine number of classes
num_classes_cic = len(np.unique(y_test_cic))
num_classes_unsw = len(np.unique(y_test_unsw))

# -------------------------------
# 📌 Convert to DMatrix
# -------------------------------

# Align feature columns exactly to what the CICIDS model expects
X_train_cic = pd.read_csv(os.path.join(dataset_dir, "X_train_cic.csv"))
cicids_feature_order = X_train_cic.columns.tolist()

X_test_cic = X_test_cic[cicids_feature_order]
X_test_unsw = X_test_unsw[cicids_feature_order]

dtest_cic = xgb.DMatrix(X_test_cic)
dtest_unsw = xgb.DMatrix(X_test_unsw)

# -------------------------------
# 📌 Load Models
# -------------------------------
print("🚀 Loading trained models...")
model_cicids = xgb.Booster()
model_cicids.load_model(model_cicids_path)

model_finetuned = xgb.Booster()
model_finetuned.load_model(model_finetuned_path)

# -------------------------------
# 📌 Make Predictions
# -------------------------------
print("🚀 Making predictions...")

# Predictions (get probabilities)
y_pred_cicids = model_cicids.predict(dtest_cic)
y_pred_finetuned_cicids = model_finetuned.predict(dtest_cic)

y_pred_cicids_unsw = model_cicids.predict(dtest_unsw)
y_pred_finetuned_unsw = model_finetuned.predict(dtest_unsw)

# Convert probabilities to class labels
y_pred_cicids = np.argmax(y_pred_cicids, axis=1)
y_pred_finetuned_cicids = np.argmax(y_pred_finetuned_cicids, axis=1)
y_pred_cicids_unsw = np.argmax(y_pred_cicids_unsw, axis=1)
y_pred_finetuned_unsw = np.argmax(y_pred_finetuned_unsw, axis=1)


print("Shape of y_test_cic:", y_test_cic.shape)
print("Shape of model_finetuned predictions:", model_finetuned.predict(dtest_cic).shape)
print("Number of classes in labels:", num_classes_cic)

# -------------------------------
# 📌 Evaluate Performance
# -------------------------------
print("\n📊 Evaluating Performance...")

# Accuracy
accuracy_cicids = accuracy_score(y_test_cic, y_pred_cicids)
accuracy_finetuned_cicids = accuracy_score(y_test_cic, y_pred_finetuned_cicids)
accuracy_cicids_unsw = accuracy_score(y_test_unsw, y_pred_cicids_unsw)
accuracy_finetuned_unsw = accuracy_score(y_test_unsw, y_pred_finetuned_unsw)

# Log Loss (now specifying labels)
logloss_cicids = log_loss(y_test_cic, model_cicids.predict(dtest_cic), labels=np.arange(num_classes_cic))
logloss_finetuned_cicids = log_loss(y_test_cic, model_finetuned.predict(dtest_cic), labels=np.arange(num_classes_cic))
# Log Loss for UNSW: Always use full 15 classes
logloss_cicids_unsw = log_loss(y_test_unsw, model_cicids.predict(dtest_unsw), labels=np.arange(15))
logloss_finetuned_unsw = log_loss(y_test_unsw, model_finetuned.predict(dtest_unsw), labels=np.arange(15))

# -------------------------------
# 📌 Print Results
# -------------------------------
print("\n📊 Model Performance Comparison:")
print(f"🔹 CICIDS Model (CICIDS Test Set) - Accuracy: {accuracy_cicids:.4f}, Log Loss: {logloss_cicids:.4f}")
print(f"🔹 Fine-Tuned Model (CICIDS Test Set) - Accuracy: {accuracy_finetuned_cicids:.4f}, Log Loss: {logloss_finetuned_cicids:.4f}")
print("\n🔹 CICIDS Model (UNSW Test Set) - Accuracy: {:.4f}, Log Loss: {:.4f}".format(accuracy_cicids_unsw, logloss_cicids_unsw))
print(f"🔹 Fine-Tuned Model (UNSW Test Set) - Accuracy: {accuracy_finetuned_unsw:.4f}, Log Loss: {logloss_finetuned_unsw:.4f}")

# -------------------------------
# 📌 Conclusion
# -------------------------------
if accuracy_finetuned_unsw > accuracy_cicids_unsw:
    print("\n✅ Fine-tuned model performed better on UNSW test set!")
else:
    print("\n⚠️ Fine-tuned model did not improve significantly on UNSW.")

print("✅ Model check complete!")
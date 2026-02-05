import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# Load the X_train data (change path if needed)
x_train = pd.read_csv("C:/Users/jahma/OneDrive/Desktop/Datasets/Processed/X_train_cic.csv")

# Fit the scaler
scaler = StandardScaler()
scaler.fit(x_train)

# Save the fitted scaler to disk
joblib.dump(scaler, "C:/Users/jahma/OneDrive/Desktop/Datasets/Processed/fitted_scaler.pkl")
print("✅ Scaler fitted and saved to C:/Users/jahma/OneDrive/Desktop/Datasets/Processed/fitted_scaler.pkl")
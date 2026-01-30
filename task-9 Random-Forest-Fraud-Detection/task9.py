# ==========================================================
# Task 9: Random Forest – Credit Card Classification
# Dataset: CreditCard.csv
# Tools: Pandas, Scikit-learn, Matplotlib
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# ----------------------------------------------------------
# Step 1: Load dataset
# ----------------------------------------------------------
df = pd.read_csv("CreditCard.csv")

print("Dataset loaded successfully!\n")
print(df.head())

# ----------------------------------------------------------
# Step 2: Target distribution
# Target column: 'card' (yes / no)
# ----------------------------------------------------------
print("\nCard Distribution:")
print(df["card"].value_counts())

# ----------------------------------------------------------
# Step 3: Encode target variable
# yes -> 1, no -> 0
# ----------------------------------------------------------
df["card"] = df["card"].map({"yes": 1, "no": 0})

# ----------------------------------------------------------
# Step 4: Drop identifier column
# ----------------------------------------------------------
if "rownames" in df.columns:
    df.drop(columns=["rownames"], inplace=True)

# ----------------------------------------------------------
# Step 5: One-Hot Encode categorical features
# (VERY IMPORTANT FIX)
# ----------------------------------------------------------
df = pd.get_dummies(df, drop_first=True)

print("\nDataset after encoding:")
print(df.head())

# ----------------------------------------------------------
# Step 6: Separate features and target
# ----------------------------------------------------------
X = df.drop("card", axis=1)
y = df["card"]

print("\nFeature matrix shape:", X.shape)
print("Target shape:", y.shape)

# ----------------------------------------------------------
# Step 7: Train-Test Split (STRATIFIED)
# ----------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ----------------------------------------------------------
# Step 8: Baseline Model – Logistic Regression
# ----------------------------------------------------------
baseline = LogisticRegression(max_iter=1000)
baseline.fit(X_train, y_train)

baseline_pred = baseline.predict(X_test)

print("\nBaseline Model (Logistic Regression) Report:")
print(classification_report(y_test, baseline_pred))

# ----------------------------------------------------------
# Step 9: Random Forest Model
# ----------------------------------------------------------
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

print("\nRandom Forest model trained successfully!")

# ----------------------------------------------------------
# Step 10: Evaluate Random Forest
# ----------------------------------------------------------
rf_pred = rf.predict(X_test)

print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_pred))

# ----------------------------------------------------------
# Step 11: Feature Importance Plot
# ----------------------------------------------------------
importances = rf.feature_importances_
indices = np.argsort(importances)[-10:]

plt.figure(figsize=(8, 5))
plt.barh(range(len(indices)), importances[indices])
plt.yticks(range(len(indices)), X.columns[indices])
plt.title("Top 10 Important Features")
plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# Step 12: Save trained model
# ----------------------------------------------------------
joblib.dump(rf, "random_forest_credit_model.pkl")

print("\nModel saved as 'random_forest_credit_model.pkl'")
print("\nTask 9 Completed Successfully!")

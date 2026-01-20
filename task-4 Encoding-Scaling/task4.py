import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

df = pd.read_csv("adult.csv")
print("Dataset loaded successfully!\n")
print(df.head())

categorical_features = [
    "workclass", "education", "marital_status",
    "occupation", "relationship", "race", "sex"
]

numerical_features = [
    "age", "education_num", "hours_per_week"
]

print("\nCategorical Features:", categorical_features)
print("Numerical Features:", numerical_features)

label_encoder = LabelEncoder()
df["education"] = label_encoder.fit_transform(df["education"])

print("\nEducation encoded using Label Encoding")

df = pd.get_dummies(
    df,
    columns=["workclass", "marital_status", "occupation", "relationship", "race", "sex"],
    drop_first=True
)

print("\nApplied One-Hot Encoding")

scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

print("\nApplied Standard Scaling to numerical features")

print("\nScaled numerical features preview:")
print(df[numerical_features].head())

df.to_csv("adult_preprocessed.csv", index=False)

print("\nPreprocessed dataset saved as 'adult_preprocessed.csv'")
print("\nTask 4 Completed Successfully!")

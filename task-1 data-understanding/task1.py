import pandas as pd
import numpy as np

df = pd.read_csv("titanic_dataset.csv")

print("Dataset Loaded Successfully!\n")

print("First 5 rows of the dataset:")
print(df.head())

print("\nLast 5 rows of the dataset:")
print(df.tail())

print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

numerical_features = ["PassengerId", "Age", "SibSp", "Parch", "Fare"]
categorical_features = ["Sex", "Embarked"]
ordinal_features = ["Pclass"]
binary_features = ["Survived"]

print("\nFeature Classification:")
print("Numerical Features:", numerical_features)
print("Categorical Features:", categorical_features)
print("Ordinal Features:", ordinal_features)
print("Binary Features:", binary_features)

print("\nUnique Values in Categorical Columns:")

for col in categorical_features:
    print(f"\n{col}:")
    print(df[col].value_counts())

target_variable = "Survived"
input_features = df.drop(columns=[target_variable]).columns.tolist()

print("\nTarget Variable:", target_variable)
print("Input Features:", input_features)

rows, columns = df.shape
print(f"\nDataset Size: {rows} rows and {columns} columns")

if rows >= 10:
    print("Dataset is suitable for basic machine learning tasks.")
else:
    print("Dataset is too small for ML.")

print("\nMissing Values in Dataset:")
print(df.isnull().sum())


print("\nData Quality Observations:")
print("- Age column contains missing values.")
print("- Dataset has both numerical and categorical data.")
print("- Target variable is binary (Survived).")
print("- Categorical features need encoding before ML modeling.")
print("- Dataset may have class imbalance.")

print("\nTask 1 Completed Successfully!")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("house_prices_dataset.csv")

print("Dataset loaded successfully!\n")

original_shape = df.shape
print("Original Dataset Shape:", original_shape)

print("\nMissing Values BEFORE cleaning:")
print(df.isnull().sum())

df.isnull().sum().plot(kind='bar', title='Missing Values Count (Before Cleaning)')
plt.xlabel("Columns")
plt.ylabel("Missing Value Count")
plt.tight_layout()
plt.show()

numerical_columns = ["LotFrontage"]

for col in numerical_columns:
    median_value = df[col].median()
    df[col].fillna(median_value, inplace=True)


categorical_columns = ["MasVnrType"]

for col in categorical_columns:
    mode_value = df[col].mode()[0]
    df[col].fillna(mode_value, inplace=True)

df.drop(columns=["PoolQC"], inplace=True)

print("\nDropped column: PoolQC (too many missing values)")

print("\nMissing Values AFTER cleaning:")
print(df.isnull().sum())

print("\nDataset Shape Comparison:")
print("Before Cleaning:", original_shape)
print("After Cleaning:", df.shape)

df.to_csv("house_prices_cleaned.csv", index=False)

print("\nCleaned dataset saved as 'house_prices_cleaned.csv'")
print("\nTask 2 Completed Successfully!")
